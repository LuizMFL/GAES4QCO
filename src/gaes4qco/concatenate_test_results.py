from pathlib import Path
from analysis.json_result_concatenator import JsonResultConcatenator


def main():
    project_root = Path(__file__).resolve().parents[2]
    tests_dir = project_root / "tests"
    results_dir = project_root / "results"

    print("🔍 Iniciando concatenação de resultados de testes...")
    concatenator = JsonResultConcatenator(tests_dir, results_dir)
    concatenated_files = concatenator.process_all_tests()

    print("\n=== ✅ CONCATENAÇÃO FINALIZADA ===")
    for path in concatenated_files:
        print(f" • {path.name}")


if __name__ == "__main__":
    main()
