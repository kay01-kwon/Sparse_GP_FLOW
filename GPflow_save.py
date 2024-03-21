import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gpflow

print(tf.config.list_physical_devices('GPU'))
print(tf.__version__)

f = lambda x: 5*np.sin(0.5*x)
x = np.arange(-5.0, 5.0, 0.1)

n_train = 1000
n_test = 2000
noise_var = 1

X = np.random.uniform(-5.0, 5.0, size=(n_train,1))

Y = f(X) + noise_var*np.random.randn(n_train).reshape(n_train,1)

Xplot = np.linspace(-5,5,n_test).reshape(-1,1)
Y_gnd = f(Xplot)

model = gpflow.models.GPR((X,Y),kernel=gpflow.kernels.SquaredExponential())

opt = gpflow.optimizers.Scipy()
opt.minimize(model.training_loss, model.trainable_variables)

f_mean, f_var = model.predict_f(Xplot, full_cov=False)
y_mean, y_var = model.predict_y(Xplot)

f_lower = f_mean - 1.96*np.sqrt(f_var)
f_upper = f_mean + 1.96*np.sqrt(f_var)
y_lower = y_mean - 1.96*np.sqrt(y_var)
y_upper = y_mean + 1.96*np.sqrt(y_var)


plt.figure(figsize=(10,5))
# plt.plot(X, Y, 'kx',mew, label='Observation')
plt.plot(Xplot,Y_gnd, color='r', label='ground truth')
plt.plot(Xplot, f_mean, '-', color='C0' ,label='mean')
plt.plot(Xplot, f_lower, '--', color='C0', label='f 95% confidence')
plt.plot(Xplot, f_upper, '--', color='C0', label='f 95% confidence')
plt.fill_between(Xplot[:,0], f_lower[:,0], f_upper[:,0], color='C0', alpha=0.1)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('GPflow Regression')
plt.legend()

plt.savefig('GPflow_tutorial.png',dpi=600)

plt.figure(figsize=(10,5))
plt.plot(X, Y, 'kx',mew=1, label='Observation')
plt.plot(Xplot, Y_gnd, color='r', label='ground truth')
plt.plot(Xplot, f_mean, '-', color='b', label='prediction')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('GPflow Regression')
plt.legend()
plt.savefig('GPflow_tutorial_obs.png',dpi=600)

# plt.show()
