import pickletools
from pathlib import Path

path1 = Path("~/Library/CloudStorage/OneDrive-UniversityofWollongong/PickleFilesForCheck/ICUstay_39553978.pkl").expanduser()

path2 = Path("~/MIMICWorkspace/MasterDataset/ICUstay_39553978.pkl").expanduser()


def check_pickle(path):
    with open(path, "rb") as f:
        data = f.read()

# Print only the lines referencing module/class lookups
    for opcode, arg, pos in pickletools.genops(data):
        if opcode.name in ("GLOBAL", "STACK_GLOBAL"):
            print(pos, arg)

check_pickle(path1)
check_pickle(path2)


