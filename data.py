from dataset import Dataset

d = Dataset("../cnn/questions/smallTest/numbered/", 3, 4000, 2000)
print d.batches

for i in range(10):
    x, y = d.next_batch()
    print x
    print y
