// ƸӜƷ Fairly Symbolic
// Authors: L. Favole, G. Mongardi, G. Squillero, A. Tonda
// Copyright 2026 INRAE / Politecnico di Torino
// SPDX-License-Identifier: MIT-0

// Macro library: loading, sampling and realizing mined macros (load_macros /
// realize / MacroLibrary::sample / arg_type). See arc/macro.hpp for the
// draw-order contract.

#include "arc/macro.hpp"

#include <cstdio>
#include <utility>

#include "arc/codec.hpp"  // TransformCodec (realize uses its arg samplers)
#include "arc/json.hpp"
#include "arc/rng.hpp"

namespace arc {

namespace {

// kind_str(node_kind): normalise a stringified kind to its lowercase enum
// value. The mined JSON stores heads as "by_color"/"translate_wrap" (already
// the enum value), but "SelectorKind.BY_COLOR" is accepted too.
std::string kind_str(const std::string& raw) {
  std::string k = raw;
  size_t dot = k.rfind('.');
  if (dot != std::string::npos) k = k.substr(dot + 1);
  for (char& c : k) {
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  }
  return k;
}

// coerce_fixed(val): recover bools from the miner's stringified categoricals.
// "True"/"False" -> Bool; else keep the string. (All mined fixed_args values
// are strings in the JSON.)
DslValue coerce_fixed(const std::string& val) {
  DslValue d;
  if (val == "True") {
    d.tag = DslValue::Tag::Bool;
    d.b = true;
  } else if (val == "False") {
    d.tag = DslValue::Tag::Bool;
    d.b = false;
  } else {
    d.tag = DslValue::Tag::Str;
    d.s = val;
  }
  return d;
}

// Update `args` with `overlay`: for each (k, v) in overlay IN ORDER, if k
// already exists in `args` overwrite its value in place (position unchanged),
// else append.
void args_update(DslArgs& args, const DslArgs& overlay) {
  for (const auto& kv : overlay) {
    bool found = false;
    for (auto& existing : args) {
      if (existing.first == kv.first) {
        existing.second = kv.second;
        found = true;
        break;
      }
    }
    if (!found) args.emplace_back(kv.first, kv.second);
  }
}

// typed_int_args(args, t): NAMES of int-scalar args in `args` whose arg_type
// is `t`, in insertion order. Bools are excluded.
std::vector<std::string> typed_int_args(const DslArgs& args,
                                        const std::string& t) {
  std::vector<std::string> out;
  for (const auto& kv : args) {
    if (kv.second.tag == DslValue::Tag::Int && macro_arg_type(kv.first) == t) {
      out.push_back(kv.first);
    }
  }
  return out;
}

// args_get_mut: pointer to the value for `key` (nullptr if absent).
DslValue* args_get_mut(DslArgs& args, const std::string& key) {
  for (auto& kv : args) {
    if (kv.first == key) return &kv.second;
  }
  return nullptr;
}

}  // namespace

std::string macro_arg_type(const std::string& name) {
  std::string n = name;
  for (char& c : n) {
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  }
  if (n.find("color") != std::string::npos) return "color";
  if (n.find("mass") != std::string::npos || n == "holes" ||
      n.find("count") != std::string::npos) {
    return "size";
  }
  return "scalar:" + n;
}

const Macro& MacroLibrary::sample(ArcRandom& rng) const {
  std::vector<int64_t> weights;
  weights.reserve(macros.size());
  for (const Macro& m : macros) weights.push_back(m.utility);
  std::vector<size_t> idx = rng.choices_indices(weights, 1);
  return macros[idx[0]];
}

std::vector<Macro> load_macros(const std::string& path) {
  std::vector<Macro> macros;

  // Read the file. Any I/O failure -> empty library (feature off).
  std::string text;
  {
    FILE* fh = std::fopen(path.c_str(), "rb");
    if (!fh) return macros;
    std::fseek(fh, 0, SEEK_END);
    long sz = std::ftell(fh);
    if (sz < 0) {
      std::fclose(fh);
      return macros;
    }
    std::fseek(fh, 0, SEEK_SET);
    text.resize(static_cast<size_t>(sz));
    size_t got = std::fread(&text[0], 1, static_cast<size_t>(sz), fh);
    std::fclose(fh);
    text.resize(got);
  }

  bool ok = false;
  JsonValue doc = json_parse(text, &ok);
  if (!ok || !doc.is_object()) return macros;

  const JsonValue* macros_arr = doc.find("macros");
  if (!macros_arr || !macros_arr->is_array()) return macros;

  for (const JsonValue& m : macros_arr->items()) {
    if (!m.is_object()) continue;

    std::vector<MacroSlot> slots;
    const JsonValue* slots_arr = m.find("slots");
    bool dropped = false;
    if (slots_arr && slots_arr->is_array()) {
      for (const JsonValue& s : slots_arr->items()) {
        if (!s.is_object()) continue;
        MacroSlot slot;

        // heads: [[sel, act, count], ...]; skip heads with unknown kinds.
        const JsonValue* heads_arr = s.find("heads");
        if (heads_arr && heads_arr->is_array()) {
          for (const JsonValue& h : heads_arr->items()) {
            if (!h.is_array() || h.size() < 3) continue;
            SelectorKind sk;
            ActionKind ak;
            DslArgs variant, act_variant;
            if (!selector_from_name(kind_str(h[0].as_string()), &sk, &variant))
              continue;
            if (!action_from_name(kind_str(h[1].as_string()), &ak, &act_variant))
              continue;
            MacroHead head;
            head.sel = sk;
            if (!variant.empty()) head.sel_variant = variant[0].second.s;
            head.act = ak;
            if (!act_variant.empty()) head.act_variant = act_variant[0].second.s;
            head.count = h[2].as_int();
            slot.heads.push_back(head);
          }
        }
        if (slot.heads.empty()) {
          // A macro with an empty slot is dropped entirely.
          slots.clear();
          dropped = true;
          break;
        }

        // role: [SelRole, ActRole] (default "?"/"?").
        const JsonValue* role_arr = s.find("role");
        std::string sel_role = "?";
        std::string act_role = "?";
        if (role_arr && role_arr->is_array() && role_arr->size() >= 2) {
          if ((*role_arr)[0].is_string()) sel_role = (*role_arr)[0].as_string();
          if ((*role_arr)[1].is_string()) act_role = (*role_arr)[1].as_string();
        }
        slot.role = {sel_role, act_role};

        // fixed_args: {"sel:name": val, "act:name": val, ...}.
        const JsonValue* fixed = s.find("fixed_args");
        if (fixed && fixed->is_object()) {
          for (const auto& kv : fixed->members()) {
            const std::string& fname = kv.first;
            const JsonValue& fval = kv.second;
            // The miner stringifies categoricals; values arrive as strings.
            std::string sval =
                fval.is_string() ? fval.as_string() : json_dump(fval);
            if (fname.rfind("sel:", 0) == 0) {
              slot.fixed_sel.emplace_back(fname.substr(4), coerce_fixed(sval));
            } else if (fname.rfind("act:", 0) == 0) {
              slot.fixed_act.emplace_back(fname.substr(4), coerce_fixed(sval));
            }
          }
        }

        slots.push_back(std::move(slot));
      }
    }
    if (dropped || slots.empty()) continue;

    Macro macro;
    const JsonValue* nm = m.find("name");
    if (nm && nm->is_string()) {
      macro.name = nm->as_string();
    } else {
      // Default name: "M" + the CURRENT macro count, zero-padded to 3 digits.
      char buf[16];
      std::snprintf(buf, sizeof(buf), "M%03zu", macros.size());
      macro.name = buf;
    }
    const JsonValue* len = m.find("length");
    macro.length = (len && len->is_number()) ? static_cast<int>(len->as_int())
                                             : static_cast<int>(slots.size());
    const JsonValue* sup = m.find("support");
    macro.support = (sup && sup->is_number()) ? static_cast<int>(sup->as_int()) : 0;
    const JsonValue* util = m.find("utility");
    int u = (util && util->is_number()) ? static_cast<int>(util->as_int()) : 1;
    macro.utility = u < 1 ? 1 : u;  // max(utility, 1)
    macro.slots = std::move(slots);

    // binding: [{"type": str, "slots": [i, j], "fraction": float}, ...].
    const JsonValue* bind_arr = m.find("binding");
    if (bind_arr && bind_arr->is_array()) {
      for (const JsonValue& b : bind_arr->items()) {
        if (!b.is_object()) continue;
        const JsonValue* bt = b.find("type");
        const JsonValue* bs = b.find("slots");
        if (!bt || !bt->is_string()) continue;
        if (!bs || !bs->is_array() || bs->size() < 2) continue;
        MacroBinding mb;
        mb.type = bt->as_string();
        mb.slot_i = static_cast<int>((*bs)[0].as_int());
        mb.slot_j = static_cast<int>((*bs)[1].as_int());
        const JsonValue* bf = b.find("fraction");
        mb.fraction = (bf && bf->is_number()) ? bf->as_double() : 1.0;
        macro.binding.push_back(mb);
      }
    }

    macros.push_back(std::move(macro));
  }
  return macros;
}

std::vector<DslRule> macro_realize(const Macro& macro, TransformCodec& codec,
                                   ArcRandom& rng) {
  // Per slot, IN ORDER: draw a head, sample sel/act args, overlay fixed args.
  struct SlotArgs {
    SelectorKind sk;
    DslArgs sel_args;
    ActionKind ak;
    DslArgs act_args;
  };
  std::vector<SlotArgs> slot_args;
  slot_args.reserve(macro.slots.size());

  for (const MacroSlot& slot : macro.slots) {
    std::vector<int64_t> weights;
    weights.reserve(slot.heads.size());
    for (const MacroHead& h : slot.heads) weights.push_back(h.count);
    std::vector<size_t> idx = rng.choices_indices(weights, 1);
    const MacroHead& head = slot.heads[idx[0]];

    SlotArgs sa;
    sa.sk = head.sel;
    sa.ak = head.act;
    sa.sel_args = codec.sample_selector_args(head.sel, rng, head.sel_variant);
    sa.act_args = codec.sample_action_args(head.act, rng);
    if (!head.act_variant.empty())
      if (DslValue* v = args_get_mut(sa.act_args, action_variant_key(head.act))) v->s = head.act_variant;
    args_update(sa.sel_args, slot.fixed_sel);
    args_update(sa.act_args, slot.fixed_act);
    slot_args.push_back(std::move(sa));
  }

  // Soft typed-hole bindings, IN ORDER.
  const int n = static_cast<int>(slot_args.size());
  for (const MacroBinding& b : macro.binding) {
    const int i = b.slot_i;
    const int j = b.slot_j;
    if (i >= n || j >= n) continue;  // out of range: NO rng draw
    if (rng.random() >= b.fraction) continue;  // soft gate (random() consumed)

    // Source: prefer sel_i (if it has a matching typed-int arg), else act_i.
    std::vector<std::string> src_sel = typed_int_args(slot_args[i].sel_args, b.type);
    std::vector<std::string> src_act = typed_int_args(slot_args[i].act_args, b.type);
    const std::vector<std::string>& src_names = !src_sel.empty() ? src_sel : src_act;
    if (src_names.empty()) continue;
    const DslArgs& src_args = !src_sel.empty() ? slot_args[i].sel_args
                                               : slot_args[i].act_args;
    // value = src_args[src_names[0]] (guaranteed present, an Int).
    DslValue value;
    for (const auto& kv : src_args) {
      if (kv.first == src_names[0]) {
        value = kv.second;
        break;
      }
    }

    // Dest: prefer sel_j; if none, act_j.
    std::vector<std::string> dst_sel = typed_int_args(slot_args[j].sel_args, b.type);
    if (!dst_sel.empty()) {
      DslValue* slot_v = args_get_mut(slot_args[j].sel_args, dst_sel[0]);
      if (slot_v) *slot_v = value;
    } else {
      std::vector<std::string> dst_act = typed_int_args(slot_args[j].act_args, b.type);
      if (!dst_act.empty()) {
        DslValue* slot_v = args_get_mut(slot_args[j].act_args, dst_act[0]);
        if (slot_v) *slot_v = value;
      }
    }
  }

  std::vector<DslRule> rules;
  rules.reserve(slot_args.size());
  for (const SlotArgs& sa : slot_args) {
    DslRule r;
    r.selector.kind = sa.sk;
    r.selector.args = sa.sel_args;
    r.action.kind = sa.ak;
    r.action.args = sa.act_args;
    rules.push_back(std::move(r));
  }
  return rules;
}

}  // namespace arc
