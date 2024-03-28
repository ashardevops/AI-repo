import numpy as np
import matplotlib.pyplot as plt

# simply here you give the following value  for pridict the house price
x_train = np.array([1.0, 2.0, 3.0])
y_train = np.array([300.0, 500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

# here you can check how many value are present in x-axis/ y-axis
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is: {m}")

# previous function also return the sam value but difficult and much od line you write
m = len(x_train)
print(f"Number of training examples is: {m}")

# here you can check the value with the help of indexis
i = 1 # Change this to 1 to see (x^1, y^1)
x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")


x_train = np.array([100.0, 200.0])
y_train = np.array([300.0, 500.0])
# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.show()


w = 100
b = 100
print(f"w: {w}")
print(f"b: {b}")




# def compute_model_output(x_train, w, b,):
#     """
#     Computes the prediction of a linear model
#     Args:
#       x (ndarray (m,)): Data, m examples 
#       w,b (scalar)    : model parameters  
#     Returns
#       y (ndarray (m,)): target values
#     """
tmp_f_wb = compute_model_output(x_train, w, b,)

# # Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# # Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# # Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()


w = 200
b = 100
x_i = 1.2
cost_1200sqft = w * x_i + b

print(f"${cost_1200sqft:.0f} thousand dollars")





import numpy as np
# %matplotlib widget
import matplotlib.pyplot as plt
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl
plt.style.use('deeplearning.mplstyle')
x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,  430,   630, 730,])
plt.close('all') 
fig, ax, dyn_items = plt_stationary(x_train, y_train)
updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)
soup_bowl()