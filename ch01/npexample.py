import numpy as np

print('1. numpy array')
x = np.array([1.0, 2.0, 3.0])
print(x)
print(type(x))

y = np.array([2.5, 4.6, 7.7])
print(x + y)
print(x - y)
print(x / y)
print(x / 2)

print('\n2. numpy multi-demesion array')
a = np.array([[1,2], [3,4], [5,6]])
print(a)
print('Note: numpy multiD array shape is: row#, col#')
print(a.shape)
print(a.dtype)

b = np.array([[0,1],[3,0],[8,0]])
print(a + b)
print(a * b)
print(a * 10)

print('\n3. numpy multi-demesion array broadcast')
c = np.array([10, 20])
print(a)
print(c)
print(a * c)

print('\n4. visit elements in multi-demesion array')
for row in a:
    print('row: ', end='')
    print(row)
    print('elements: ', end='')
    for el in row:
        print(el, end=', ')
    print('')

print('\n5. visit elements using list')
A = a.flatten()
print(A);
print(A[np.array([0, 2, 4])])
print(A > 3)
print(A[A>3])
