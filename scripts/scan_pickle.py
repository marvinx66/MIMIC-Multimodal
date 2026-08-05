import pickletools

from pathlib import Path

def find_global_refs(path):
    with open(path, "rb") as f:
        data = f.read()

    refs = []
    string_stack = []  # tracks recent string-push opcodes

    for opcode, arg, pos in pickletools.genops(data):
        if opcode.name == "GLOBAL":
            # old-style: arg is "module class" directly
            refs.append((pos, arg))
        elif opcode.name == "STACK_GLOBAL":
            # module/class name were pushed as the last two strings
            if len(string_stack) >= 2:
                module, name = string_stack[-2], string_stack[-1]
                refs.append((pos, f"{module} {name}"))
            else:
                refs.append((pos, "UNKNOWN (stack too short)"))
        elif opcode.name in (
            "SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8",
            "SHORT_BINSTRING", "BINSTRING",
        ):
            string_stack.append(arg)
            # keep it bounded
            if len(string_stack) > 10:
                string_stack.pop(0)

    return refs

if __name__ == "__main__":

    '''
    python scan_pickle.py > path1_refs.txt
    python scan_pickle.py > path2_refs.txt    
    '''
    # import sys
    # path = sys.argv[1]
    path1 = Path("~/Library/CloudStorage/OneDrive-UniversityofWollongong/PickleFilesForCheck/ICUstay_39553978.pkl").expanduser()
    path2 = Path("~/MIMICWorkspace/MasterDataset/ICUstay_39553978.pkl").expanduser()
    refs = find_global_refs(path1)
    refs = find_global_refs(path2)
    for pos, ref in refs:
        print(pos, ref)