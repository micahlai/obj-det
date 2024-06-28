import matplotlib.pyplot as plt

x = [1,2,3,4]
y1 = [1,2,3,4]
y2 = [2,3,4,1]
labels = ['I', 'give', 'zero', 'fucks']

figure, axis = plt.subplots(2,2)

axis[0,0].plot(x,y1,c='Red')
axis[0,0].set_title("RED")
axis[0,1].plot(x,y2,c='Blue')
axis[0,1].set_title("Blue")
for i, txt in enumerate(labels):
    axis[0,1].annotate(txt, (x[i], y2[i]))

plt.savefig('destination %s.eps' % str(5), format='eps')