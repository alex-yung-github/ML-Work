total = []
with open("text.txt", "r") as f:
    for line in f:
        temp = line.split(" ")
        val = temp[-1].strip()
        # val = int(val[1:])
        total.append(val)

yessir = total.sort()
print(total)