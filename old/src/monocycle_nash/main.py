"""monocycle_nash - 単相性モデルのナッシュ均衡ソルバー"""

from monocycle_nash.presentation.cli import main as cli_main


def main() -> int:
    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
