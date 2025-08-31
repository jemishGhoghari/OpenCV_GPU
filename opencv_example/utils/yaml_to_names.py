import re, sys, pathlib

if len(sys.argv) != 3:
    print("Usage: python yaml_to_names.py coco.yaml coco.names"); sys.exit(1)

yaml = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")

names = []

# Try inline map: names: {0: person, 1: bicycle, ...}
m = re.search(r"names\s*:\s*\{([^\}]*)\}", yaml, re.S)
if m:
    inside = m.group(1)
    for tok in inside.split(","):
        tok = tok.strip()
        if ":" in tok:
            idx, val = tok.split(":", 1)
            idx = int(idx.strip())
            val = val.strip().strip("\"'")  # remove quotes
            while len(names) <= idx: names.append("")
            names[idx] = val

# If empty, try multiline map under "names:"
if not any(names):
    after = yaml.split("names:", 1)[1]
    for line in after.splitlines()[1:]:
        t = line.strip()
        if not t: continue
        if not line[0].isspace() and ":" in t and not t.startswith("-"):
            break  # next top-level key
        m = re.match(r"^(\d+)\s*:\s*(.+?)\s*(#.*)?$", t)
        if m:
            idx = int(m.group(1))
            val = m.group(2).strip().strip("\"'")
            while len(names) <= idx: names.append("")
            names[idx] = val
        else:
            m = re.match(r"^-\s*(.+?)\s*(#.*)?$", t)
            if m:
                names.append(m.group(1).strip().strip("\"'"))

# Compact and validate
names = [n for n in names if n != ""]
if not names:
    print("No names found in YAML."); sys.exit(2)

pathlib.Path(sys.argv[2]).write_text("\n".join(names) + "\n", encoding="utf-8")
print(f"Wrote {len(names)} names to {sys.argv[2]}")