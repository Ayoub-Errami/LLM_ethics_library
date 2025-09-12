import os

from prompt_wrapper import OutputComponentType, DecisionOption, PromptWrapper

from prompt_factory import get_all_possible_prompts, base_prompts
from prompt_wrapper import OutputComponentType, PromptWrapper
from prompts_json import generate_prompt_json
from version import VERSION


prompts_folder_path = os.path.join(os.path.dirname(__file__), "../data/prompts")


def generate_promopts_v1_6():
    # Generate wrapped prompts for - v1.6
    prompts = get_all_possible_prompts()
    print(f"There are a total of {len(prompts)} prompts before filtering")
    print(f"Performing filtering...")

    # Selected dilemmas
    prompts: list[PromptWrapper] = [x for x in prompts if x.dilemma.context_identifier in [
        "child_abuse_prevention",
        "public_health",
        "trolley_problem",
        "surveillance",
    ]]

    # We found the base_prompt not to have a significant impact, so we will only use base_prompt_1
    selected_base_prompt = next(iter(base_prompts))
    prompts: list[PromptWrapper] = [x for x in prompts if x.base_prompt_identifier == selected_base_prompt]

    # Our current dilemmas do not include the LLM as a subject, thus the egoism theory might not applicable
    # prompts: list[PromptWrapper] = [x for x in prompts if not x.normative_ethical_theory == "ethical_egoism"]

    # This is a newly added variable we do not yet want to test
    # Previously the prompt always containted the output structure json and description
    prompts: list[PromptWrapper] = [x for x in prompts if x.prompt_has_output_structure_description is True]
    prompts: list[PromptWrapper] = [x for x in prompts if x.prompt_has_output_structure_json_schema is True]

    print(f"Nach dem Filtern sind {len(prompts)} Prompts übrig.")

    prompts_file_path = os.path.join(prompts_folder_path, f"wrapped_prompts_v{VERSION}.json")
    generate_prompt_json(prompts, prompts_file_path)



def generate_promopts_v1_7():
    # Generate wrapped prompts for - v1.7
    prompts = get_all_possible_prompts()
    print(f"There are a total of {len(prompts)} prompts before filtering")
    print(f"Performing filtering...")

    # Selected dilemmas
    prompts: list[PromptWrapper] = [x for x in prompts if x.dilemma.context_identifier in [
        "child_abuse_prevention",
        "public_health",
        "trolley_problem",
        "surveillance",
    ]]

    # prompts: list[PromptWrapper] = [x for x in prompts if x.dilemma.identifier in [
    #     # "public_health_1",
    #     # "public_health_2",
    #     # "public_health_3",
    #     # "public_health_4",
    #     # "child_abuse_prevention_3",
    #     # "child_abuse_prevention_4",
    # ]]
    # We found the base_prompt not to have a significant impact, so we will only use base_prompt_1
    selected_base_prompt = next(iter(base_prompts))
    prompts: list[PromptWrapper] = [x for x in prompts if x.dilemma.identifier in ["public_health_1", "public_health_2"] or x.base_prompt_identifier == selected_base_prompt]

    # Our current dilemmas do not include the LLM as a subject, thus the egoism theory might not applicable
    # prompts: list[PromptWrapper] = [x for x in prompts if not x.normative_ethical_theory == "ethical_egoism"]

    # This is a newly added variable we do not yet want to test
    # Previously the prompt always containted the output structure json and description
    prompts: list[PromptWrapper] = [x for x in prompts if x.prompt_has_output_structure_description is True]
    prompts: list[PromptWrapper] = [x for x in prompts if x.prompt_has_output_structure_json_schema is True]

    # prompts: list[PromptWrapper] = [x for x in prompts if not x.output_structure.get_has_output_component(OutputComponentType.NORMATIVE_ETHICAL_THEORY_EXPLANATION) is True]

    print(f"Nach dem Filtern sind {len(prompts)} Prompts übrig.")

    prompts_file_path = os.path.join(prompts_folder_path, f"wrapped_prompts_v{VERSION}.json")
    generate_prompt_json(prompts, prompts_file_path)




def generate_570_prompts_v1_8():
    """
    Auswahl für den Modellvergleich:
      - Alle 19 Dilemmas (kein Kontext-Filter)
      - Base Prompt: nur 'base_prompt_1'
      - FUO: False (kein first_unstructured_output)
      - Theorien: alle 5 (inkl. 'decide_yourself')
      - Output:
          (a) DECISION-only
          (b) ALL-THREE, aber NUR in fester Reihenfolge: Decision -> Reason -> Explanation
      - Enum-Order: 3 (YES-first, NO-first, UNDEC-first)
      - Strukturhinweise: True/True
    """
    import os
    from prompt_factory import get_all_possible_prompts
    from prompt_wrapper import OutputComponentType, DecisionOption, PromptWrapper
    from prompts_json import generate_prompt_json

    prompts = get_all_possible_prompts()
    print(f"[all19_fixedorder] total before filtering: {len(prompts)}")

    # 1) Base-Prompt: exakt 1
    SELECTED_BASE = "base_prompt_1"
    prompts = [p for p in prompts if p.base_prompt_identifier == SELECTED_BASE]
    print(f"[all19_fixedorder] after base_prompt={SELECTED_BASE}: {len(prompts)}")

    # 2) Strukturhinweise: beide an
    prompts = [p for p in prompts if p.prompt_has_output_structure_description and p.prompt_has_output_structure_json_schema]
    print(f"[all19_fixedorder] after structure-hints True/True: {len(prompts)}")

    # 3) FUO: nur False
    prompts = [p for p in prompts if not p.output_structure.first_unstructured_output]
    print(f"[all19_fixedorder] after FUO=False: {len(prompts)}")

    # 4) Output-Komponentensets:
    #    (a) DECISION-only
    def is_decision_only(p: PromptWrapper) -> bool:
        os_ = p.output_structure
        return (
            os_.get_has_output_component(OutputComponentType.DECISION)
            and not os_.get_has_output_component(OutputComponentType.NORMATIVE_ETHICAL_THEORY_EXPLANATION)
            and not os_.get_has_output_component(OutputComponentType.DECISION_REASON)
        )

    #    (b) ALL-THREE mit fester Reihenfolge: Decision -> Reason -> Explanation
    def is_all_three_fixed(p: PromptWrapper) -> bool:
        os_ = p.output_structure
        has_all = (
            os_.get_has_output_component(OutputComponentType.NORMATIVE_ETHICAL_THEORY_EXPLANATION)
            and os_.get_has_output_component(OutputComponentType.DECISION_REASON)
            and os_.get_has_output_component(OutputComponentType.DECISION)
        )
        if not has_all:
            return False
        fixed_order = [
            OutputComponentType.DECISION,
            OutputComponentType.DECISION_REASON,
            OutputComponentType.NORMATIVE_ETHICAL_THEORY_EXPLANATION,
        ]
        # In deiner Lib heißt das Feld 'sorted_output_components'
        return os_.sorted_output_components == fixed_order

    prompts = [p for p in prompts if is_decision_only(p) or is_all_three_fixed(p)]
    print(f"[all19_fixedorder] after component-set (DECISION-only or ALL-THREE fixed): {len(prompts)}")

    # 5) Enum-Order: 3 Ordnungen
    KEEP_ENUMS = {
        (DecisionOption.YES, DecisionOption.NO, DecisionOption.UNDECIDED),   # YES first
        (DecisionOption.NO, DecisionOption.YES, DecisionOption.UNDECIDED),   # NO first
        (DecisionOption.UNDECIDED, DecisionOption.YES, DecisionOption.NO),   # UNDEC first
    }
    prompts = [p for p in prompts if tuple(p.output_structure.sorted_decision_options) in KEEP_ENUMS]
    print(f"[all19_fixedorder] after enum-orders (3): {len(prompts)}")

    # 6) Sortieren & Schreiben
    prompts.sort(key=lambda x: x._id)
    out_dir = os.path.join(os.path.dirname(__file__), "../data/prompts")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "wrapped_prompts_570_v1.8.json")
    generate_prompt_json(prompts, out_path)
    print(f"[all19_fixedorder] wrote {len(prompts)} prompts to {out_path}")


def generate_3420_prompts_v1_8():
    """
    Erzeugt das vollständige 3,420er-Set:
      - Base Prompts: base_prompt_1, base_prompt_2, base_prompt_3
      - Strukturhinweise: True/True (Beschreibung + JSON Schema)
      - Decision-Enum-Order: Genau die 3 Permutationen (YES-first, NO-first, UNDEC-first)
      - Output-Komponenten: entweder
          (a) DECISION-only, oder
          (b) ALL-THREE in fester Reihenfolge: Decision -> Reason -> Explanation
      - FUO: egal (kein Filter)
    """
    import os
    from pathlib import Path
    from prompt_factory import get_all_possible_prompts
    from prompt_wrapper import OutputComponentType, DecisionOption, PromptWrapper
    from prompts_json import generate_prompt_json

    KEEP_BASES = {"base_prompt_1", "base_prompt_2", "base_prompt_3"}
    KEEP_ENUMS = {
        (DecisionOption.YES, DecisionOption.NO, DecisionOption.UNDECIDED),   # YES first
        (DecisionOption.NO, DecisionOption.YES, DecisionOption.UNDECIDED),   # NO first
        (DecisionOption.UNDECIDED, DecisionOption.YES, DecisionOption.NO),   # UNDEC first
    }
    FIXED_ALLTHREE_ORDER = [
        OutputComponentType.DECISION,
        OutputComponentType.DECISION_REASON,
        OutputComponentType.NORMATIVE_ETHICAL_THEORY_EXPLANATION,
    ]

    def is_hints_on(p: PromptWrapper) -> bool:
        return p.prompt_has_output_structure_description and p.prompt_has_output_structure_json_schema

    def is_enum_kept(p: PromptWrapper) -> bool:
        return tuple(p.output_structure.sorted_decision_options) in KEEP_ENUMS

    def is_decision_only(p: PromptWrapper) -> bool:
        os_ = p.output_structure
        return (
            os_.get_has_output_component(OutputComponentType.DECISION)
            and not os_.get_has_output_component(OutputComponentType.DECISION_REASON)
            and not os_.get_has_output_component(OutputComponentType.NORMATIVE_ETHICAL_THEORY_EXPLANATION)
        )

    def is_all_three_fixed(p: PromptWrapper) -> bool:
        os_ = p.output_structure
        has_all = (
            os_.get_has_output_component(OutputComponentType.DECISION)
            and os_.get_has_output_component(OutputComponentType.DECISION_REASON)
            and os_.get_has_output_component(OutputComponentType.NORMATIVE_ETHICAL_THEORY_EXPLANATION)
        )
        if not has_all:
            return False
        # robust ggü. evtl. fehlendem Feld
        try:
            return os_.sorted_output_components == FIXED_ALLTHREE_ORDER
        except Exception:
            return False

    # ---- Generieren & Filtern
    prompts = get_all_possible_prompts()
    print(f"[all19_3420] total before filtering: {len(prompts)}")

    prompts = [
        p for p in prompts
        if (p.base_prompt_identifier in KEEP_BASES)
        and is_hints_on(p)
        and is_enum_kept(p)
        and (is_decision_only(p) or is_all_three_fixed(p))
    ]
    print(f"[all19_3420] after 3420-design filter: {len(prompts)}  (sollte 3420 sein)")

    # Stabil sortieren und schreiben
    prompts.sort(key=lambda x: x._id)
    out_dir = Path(os.path.join(os.path.dirname(__file__), "../data/prompts"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wrapped_prompts_3420_v1.8.json"
    generate_prompt_json(prompts, str(out_path))
    print(f"[all19_3420] wrote {len(prompts)} prompts to {out_path}")
    return str(out_path)


if __name__ == "__main__":
    generate_570_prompts_v1_8()

