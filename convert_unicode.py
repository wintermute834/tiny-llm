
import sys

from unidecode import unidecode



if len(sys.argv) > 1:
    path = sys.argv[1]
    
    with open(path, "r") as f:
        text = f.read()

    with open(path, "w") as f:
        f.write(unidecode(text))

    print("Converted")

