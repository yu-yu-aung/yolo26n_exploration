from ultralytics import YOLO
import inspect

model = YOLO("models/yolo26n.pt")
criterion = model.model.init_criterion()

print("=" * 70)
print("E2ELoss IMPLEMENTATION")
print("=" * 70)

print(inspect.getsource(type(criterion)))

print("\n" + "=" * 70)
print("LOSS COMPONENTS")
print("=" * 70)

for name in ["one2many", "one2one", "o2m", "o2o", "total"]: 
  if hasattr(criterion, name): 
    component = getattr(criterion, name)
    print(f"\n{name}:")
    print("Type:", type(component))
    print(component)