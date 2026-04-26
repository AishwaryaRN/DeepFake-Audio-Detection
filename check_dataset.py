import os

base_path = r"dataset\LA\ASVspoof2019_LA_cm_protocols"

if not os.path.exists(base_path):
    print("Folder not found!")
else:
    print("Protocol files found:\n")
    for file in os.listdir(base_path):
        print(file)