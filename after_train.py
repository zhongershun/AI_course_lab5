import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

alllosses = np.loadtxt("./DataLog/alllosses.csv")
loss_pre_epoch = np.loadtxt("./DataLog/loss_pre_epoch.csv")
vallosses = np.loadtxt("./DataLog/vallosses.csv")
acces = np.loadtxt("./DataLog/acces.csv")
# print(alllosses)


plt.plot([i for i in range(len(loss_pre_epoch))], loss_pre_epoch)
plt.title("loss for each epoch")
plt.savefig("./figout/loss_per_epoch.png")
plt.show()

plt.plot(alllosses)
plt.title("loss on train_set during training processing")
plt.savefig("./figout/train_loss.png")
plt.show()

plt.plot(vallosses)
plt.title("loss on valid_set during training processing")
plt.savefig("./figout/valid_loss.png")
plt.show()

plt.plot(acces)
plt.title('accuracy')
plt.savefig("./figout/acc.png")
plt.show()