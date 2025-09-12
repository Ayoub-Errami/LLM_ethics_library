import os
import time
import datetime
import argparse
import sys
from prompts_json import load_prompts_from_json, generate_response_json
from open_ai_wrapper import query_openai_api
from mistral_wrapper import query_mistral_api
from deepseek_wrapper import query_deepseek_api
from prompt_wrapper import LlmName

def format_time(seconds: float) -> str:
    if seconds is None or seconds != seconds:  # NaN
        return "--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["openai", "mistral", "deepseek"], required=True,
                        help="Welches Modell soll abgefragt werden?")
    parser.add_argument("--prompts", default="../data/prompts/wrapped_prompts_3420_v1.8.json",
                        help="Pfad zur Prompts-JSON")
    parser.add_argument("--version", default="v1.8",
                        help="Versionsordner unter ../data/results (Default: v1.8)")
    parser.add_argument("--tag", default="", help="Optionaler Suffix für den Dateinamen")
    parser.add_argument("--batch_size", type=int, default=200, help="Batch-Größe zum Zwischenspeichern")
    parser.add_argument("--no_all_file", action="store_true",
                        help="Kein Gesamtfile am Ende schreiben (nur Batches).")
    args = parser.parse_args()
    model = args.model

    # 1) Prompts laden
    prompts = load_prompts_from_json(args.prompts)
    prompts = prompts[:1200]
    total = len(prompts)
    print(f"Geladen: {total} Prompts")

    # 2) Modell auswählen
    if model == "openai":
        func = query_openai_api
        llm_enum = LlmName.GPT4O
        key = os.environ.get("ETHICS_OPENAI_API_KEY")
        if not key:
            raise ValueError("Bitte setze die Umgebungsvariable ETHICS_OPENAI_API_KEY für OpenAI.")
    elif model == "mistral":
        func = query_mistral_api
        llm_enum = LlmName.MISTRAL
        key = "not-needed"
    elif model == "deepseek":
        func = query_deepseek_api
        llm_enum = LlmName.DEEPSEEK
        key = "not-needed"
    else:
        raise ValueError(f"Unbekanntes Modell: {model}")

    # 3) Pfade vorbereiten
    ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    tag = f"_{args.tag}" if args.tag else ""
    base_dir = os.path.join("..", "data", "results", args.version, model)
    os.makedirs(base_dir, exist_ok=True)

    # 4) Lauf mit Live-Progress + Batch-Speicherungen
    responses_all = []
    start_time = time.time()
    last_update_len = 0
    done = 0
    batch_size = max(1, args.batch_size)

    def print_progress(done_count: int):
        nonlocal last_update_len
        now = time.time()
        elapsed = now - start_time
        avg = elapsed / done_count if done_count > 0 else 0.0
        remain = (total - done_count) * avg if done_count > 0 else 0.0
        pct = (done_count / total * 100.0) if total else 100.0
        line = (f"[{model}] {done_count}/{total} ({pct:5.1f}%) | "
                f"avg/it: {avg:5.2f}s | elapsed: {format_time(elapsed)} | "
                f"ETA: {format_time(remain)}")
        sys.stdout.write("\r" + line + " " * max(0, last_update_len - len(line)))
        sys.stdout.flush()
        last_update_len = len(line)

    def save_batch(batch_responses, batch_idx, start_idx, end_idx_inclusive):
        # Speichere NUR die Batch-Responses als eigenes File
        batch_path = os.path.join(
            base_dir,
            f"{ts}_{model}_result_batch_{start_idx:05d}-{end_idx_inclusive:05d}{tag}.json"
        )
        generate_response_json(batch_responses, batch_path)
        print(f"\n Batch {batch_idx}: {batch_path} ({len(batch_responses)} Einträge)")

    try:
        batch_idx = 1
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_responses = []

            for i in range(start, end):
                p = prompts[i]
                try:
                    resp = func(key, p, model=llm_enum)
                except Exception as e:
                    resp = {"error": str(e), "prompt_id": getattr(p, "_id", None)}
                responses_all.append(resp)
                batch_responses.append(resp)

                done += 1
                print_progress(done)

            # Nach JEDEM Batch abspeichern
            save_batch(
                batch_responses=batch_responses,
                batch_idx=batch_idx,
                start_idx=start,
                end_idx_inclusive=end - 1
            )
            batch_idx += 1

        sys.stdout.write("\n")
        sys.stdout.flush()

    except KeyboardInterrupt:
        # Falls der Lauf abgebrochen wird, letzten (teilweisen) Batch sichern
        sys.stdout.write("\n\n KeyboardInterrupt – sichere aktuellen Fortschritt...\n")
        sys.stdout.flush()
        # Ermitteln, wie viele in der letzten (unvollständigen) Batch sind:
        already_full = (done // batch_size) * batch_size
        leftover = done - already_full
        if leftover > 0:
            # Speichere die letzten 'leftover' Antworten als Notfall-Batch
            start_idx = already_full
            end_idx_inclusive = already_full + leftover - 1
            emergency_batch = responses_all[-leftover:]
            generate_response_json(
                emergency_batch,
                os.path.join(
                    base_dir,
                    f"{ts}_{model}_result_batch_{start_idx:05d}-{end_idx_inclusive:05d}{tag}_INCOMPLETE.json"
                )
            )
            print(f" Notfall-Batch gespeichert: {leftover} Antworten")
        raise

    # 5) Optional: zusätzlich ein Gesamtfile schreiben
    total_elapsed = time.time() - start_time
    if not args.no_all_file:
        all_path = os.path.join(base_dir, f"{ts}_{model}_result_ALL{tag}.json")
        generate_response_json(responses_all, all_path)
        print(f" Gesamtfile: {all_path}")

    print(f" Gesamtdauer: {format_time(total_elapsed)} ({total} Prompts, {model})")

if __name__ == "__main__":
    main()
