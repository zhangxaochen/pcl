import numpy as np
import transformations as tf
%pylab

'一些别名'
tf.e2m = tf.euler_matrix
tf.m2e = tf.euler_from_matrix
tf.q2m = tf.quaternion_matrix
tf.m2q = tf.quaternion_from_matrix
tf.e2q = tf.quaternion_from_euler
tf.q2e = tf.euler_from_quaternion
tf.m2r = tf.rotation_from_matrix
tf.r2m = tf.rotation_matrix

rtGtFn = r'syntheticRT_quaternion.txt'
rtKfFn = r'rotateY_onecube_quaternion_matrix.csv'
rtGt = np.loadtxt(rtGtFn, delimiter=' ')
rtKf = np.loadtxt(rtKfFn, delimiter=',')

# 没用过：
tGt = rtGt[:3]
rGt = rtGt[3:]
tKf = rtKf[:3]
rKf = rtKf[3:]

# q: wxyz -> euler angle, 注意文件中是 wxyz 序， tf 用 xyzw 序
# 孙国飞输出是 sxyz？ 之前是 rxyz。	测试结果仍是 rxyz
eulGt = [tf.q2e(list(r[4:])+[r[3]]) for r in rtGt]
eulGtrxyz = [tf.q2e(list(r[4:])+[r[3]], 'rxyz') for r in rtGt]

plot(eulGt)
figure()
plot(eulGtrxyz)

#syntheticRT.txt 现在是 12 列，妈的：
rtGtFn2 = r'syntheticRT.txt'
rtGt2 = np.loadtxt(rtGtFn2, delimiter=' ')
eulGt_m2e=[tf.m2e(r[3:].reshape(3, -1)) for r in rtGt2]
eulGt_m2e_rxyz=[tf.m2e(r[3:].reshape(3, -1), 'rxyz') for r in rtGt2]

figure()
plot(eulGt_m2e)
figure()
plot(eulGt_m2e_rxyz)


# 假定 KinFu 用的 sxyz, 不确定？
eulKf = [tf.q2e(list(r[4:])+[r[3]]) for r in rtKf]
eulKfrxyz = [tf.q2e(list(r[4:])+[r[3]], 'rxyz') for r in rtKf]
figure()
plot(eulKf)
figure()
plot(eulKfrxyz)
