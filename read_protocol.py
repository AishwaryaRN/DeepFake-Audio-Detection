protocol = r"dataset\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"

labels = {}

with open(protocol, "r") as f:
    for line in f:
        parts = line.strip().split()
        file_id = parts[1]          # second column = audio file
        label = parts[-1]           # last column = bonafide/spoof
        
        labels[file_id] = 0 if label == "bonafide" else 1

print("Total files:", len(labels))
print("Bonafide:", list(labels.values()).count(0))
print("Spoof:", list(labels.values()).count(1))