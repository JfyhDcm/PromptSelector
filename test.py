from model import *

model = PModel()
for name, child in model.named_children():
    print(name)
