import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from webnovel_kb.server import kb


def batch_ingest(novels_dir: str, catalog_file: str = ""):
    novels_path = Path(novels_dir)
    if not novels_path.exists():
        print(f"目录不存在: {novels_dir}")
        return

    catalog = {}
    if catalog_file and Path(catalog_file).exists():
        with open(catalog_file, "r", encoding="utf-8") as f:
            catalog = json.load(f)

    txt_files = list(novels_path.glob("*.txt"))
    if not txt_files:
        print(f"未找到txt文件: {novels_dir}")
        return

    print(f"找到 {len(txt_files)} 个txt文件")
    for i, f in enumerate(txt_files, 1):
        stem = f.stem
        info = catalog.get(stem, {})
        title = info.get("title", stem)
        author = info.get("author", "未知")
        genre = info.get("genre", "未分类")
        file_size_mb = f.stat().st_size / (1024 * 1024)
        print(f"[{i}/{len(txt_files)}] 导入: {title} ({author}) [{genre}] {file_size_mb:.1f}MB")
        start = time.time()
        result = kb.ingest_novel(str(f), title, author, genre)
        elapsed = time.time() - start
        if "error" in result:
            print(f"  ❌ 失败: {result['error']}")
        else:
            print(f"  ✅ 成功: {result['chunk_count']} 个分块, 耗时 {elapsed:.1f}s")

    stats = kb.get_stats()
    print(f"\n导入完成！总计: {stats['total_novels']} 本小说, {stats['total_chunks']} 个分块")


def create_catalog_template(novels_dir: str, output_file: str = ""):
    novels_path = Path(novels_dir)
    if not novels_path.exists():
        print(f"目录不存在: {novels_dir}")
        return

    out = output_file or str(novels_path / "catalog.json")
    catalog = {}
    for f in novels_path.glob("*.txt"):
        catalog[f.stem] = {
            "title": f.stem,
            "author": "请填写作者",
            "genre": "请填写类型（如：奇幻/悬疑/赛博朋克/修仙/都市）"
        }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    print(f"目录模板已生成: {out}")
    print("请编辑此文件填写每本小说的作者和类型，然后运行 batch_ingest")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="网文知识库批量导入工具")
    sub = parser.add_subparsers(dest="command")

    ingest_parser = sub.add_parser("ingest", help="批量导入小说")
    ingest_parser.add_argument("novels_dir", help="小说txt文件所在目录")
    ingest_parser.add_argument("--catalog", default="", help="目录信息JSON文件")

    catalog_parser = sub.add_parser("catalog", help="生成目录模板")
    catalog_parser.add_argument("novels_dir", help="小说txt文件所在目录")
    catalog_parser.add_argument("--output", default="", help="输出文件路径")

    args = parser.parse_args()
    if args.command == "ingest":
        batch_ingest(args.novels_dir, args.catalog)
    elif args.command == "catalog":
        create_catalog_template(args.novels_dir, args.output)
    else:
        parser.print_help()
