from ultralytics import YOLO
import inspect

# Load model
model = YOLO("models/yolo26n.pt")
pytorch_model = model.model


# Inspect init_criterion()
print("=" * 70)
print("INIT CRITERION")
print("=" * 70)

#use getsource to retrieve the source code of an object when that source is available 

print(inspect.getsource(pytorch_model.init_criterion))  #init_criterion initializes and returns the criterions used to calculate the training loss
                                                        #in ML, a criterion is a function or object used to calculate the loss
                                                        #loss measures how far the model's predictions are from the expected targets 


# Create criterion
print("\n" + "=" * 70)
print("CRITERION")
print("=" * 70)

criterion = pytorch_model.init_criterion()

print("Criterion object:")
print(criterion)

print("\nCriterion type:")
print(type(criterion))


# Criterion attributes
print("\n" + "=" * 70)
print("CRITERION ATTRIBUTES")
print("=" * 70)

for name in dir(criterion):         #dir(criterion) returns a list of names associated with the object 
    if not name.startswith("_"):
        print(name)