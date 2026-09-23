"""CLI: the same operations as the MCP server, for agents that run commands.

    python -m bepic_worlds create ref.jpg --name valley --spec "misty pine forest"
    python -m bepic_worlds list
    python -m bepic_worlds describe valley
    python -m bepic_worlds edit valley '[{"op": "scale_scatter", "factor": 0.5}]' --note "fewer trees"
    python -m bepic_worlds feedback valley [--status all]
    python -m bepic_worlds resolve valley fb_1a2b3c4d --reply "thinned the pines"
    python -m bepic_worlds revert valley 2
    python -m bepic_worlds open valley
    python -m bepic_worlds schema
    python -m bepic_worlds mcp                  # run the MCP server over stdio

Output is JSON. --url / --root / --mode choose the backend (see client.py).
"""

import argparse
import json
import sys

from . import client


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bepic_worlds", description="Walkable worlds from reference images.")
    ap.add_argument("--url", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--mode", choices=["auto", "http", "local"], default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("create")
    c.add_argument("reference")
    c.add_argument("--name", default="world")
    c.add_argument("--spec", default="")
    c.add_argument("--depth"); c.add_argument("--heightmap"); c.add_argument("--panorama")
    c.add_argument("--fov", type=float, default=50.0)
    c.add_argument("--world-size", type=float, default=240.0)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--overwrite", action="store_true")
    c.add_argument("--no-open", action="store_true")

    sub.add_parser("list")
    d = sub.add_parser("describe"); d.add_argument("name")
    j = sub.add_parser("json"); j.add_argument("name"); j.add_argument("--version", type=int)
    e = sub.add_parser("edit"); e.add_argument("name"); e.add_argument("ops")
    e.add_argument("--note", default=""); e.add_argument("--no-open", action="store_true")
    r = sub.add_parser("revert"); r.add_argument("name"); r.add_argument("version", type=int)
    f = sub.add_parser("feedback"); f.add_argument("name"); f.add_argument("--status", default="open")
    rs = sub.add_parser("resolve"); rs.add_argument("name"); rs.add_argument("ids", nargs="+")
    rs.add_argument("--reply", default="")
    o = sub.add_parser("open"); o.add_argument("name"); o.add_argument("--version", type=int)
    sub.add_parser("schema")
    sub.add_parser("mcp")
    args = ap.parse_args(argv)

    if args.cmd == "mcp":
        from . import mcp_server
        rest = [x for pair in (("--url", args.url), ("--root", args.root), ("--mode", args.mode))
                for x in pair if pair[1]]
        return mcp_server.main(rest)
    if args.cmd == "schema":
        print(client.schema_text())
        return 0

    be = client.backend(url=args.url, root=args.root, mode=args.mode)
    try:
        if args.cmd == "create":
            out = be.create(args.reference, name=args.name, spec=args.spec or None, depth=args.depth,
                            heightmap=args.heightmap, panorama=args.panorama, fov=args.fov,
                            world_size=args.world_size, seed=args.seed, overwrite=args.overwrite,
                            open_in_viewer=not args.no_open)
        elif args.cmd == "list":
            out = be.list()
        elif args.cmd == "describe":
            out = be.describe(args.name)
        elif args.cmd == "json":
            out = be.world_json(args.name, args.version)
        elif args.cmd == "edit":
            out = be.edit(args.name, json.loads(args.ops), note=args.note, open_in_viewer=not args.no_open)
        elif args.cmd == "revert":
            out = be.revert(args.name, args.version)
        elif args.cmd == "feedback":
            out = be.feedback(args.name, args.status)
        elif args.cmd == "resolve":
            out = be.resolve(args.name, args.ids, reply=args.reply)
        elif args.cmd == "open":
            out = be.open(args.name, args.version)
    except Exception as e:
        print(json.dumps({"error": str(e), "backend": be.kind}), file=sys.stderr)
        return 1
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
