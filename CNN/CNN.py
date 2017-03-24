import tensorflow as tf
import GetFeatures as gf
import GenPlates as gp
import numpy as np

logs_path = '/tmp/tensorflow_logs/example'
CHARS = gf.CHARS

images,plates=gf.readData(gf.dataPath)
plates=gf.formatPlates(plates)
test_x,test_y=gf.getTestData(images,plates)

n_classes = 7*len(CHARS)
batch_size = 100
dropout=0.75

x = tf.placeholder('float', [None,80*160*3])
y = tf.placeholder('float', [None, 7*len(CHARS)])
keep_prob=tf.placeholder('float')

weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,3,48])),
           'W_conv2':tf.Variable(tf.random_normal([5,5,48,64])),
           'W_conv3':tf.Variable(tf.random_normal([5,5,64,128])),
           'W_fc1':tf.Variable(tf.random_normal([10*20*128,1024])),
           'W_fc2':tf.Variable(tf.random_normal([1024,2048])),
           'out':tf.Variable(tf.random_normal([2048,n_classes])),}

biases = {'b_conv1':tf.Variable(tf.random_normal([48])),
          'b_conv2':tf.Variable(tf.random_normal([64])),
          'b_conv3':tf.Variable(tf.random_normal([128])),
          'b_fc1':tf.Variable(tf.random_normal([1024])),
          'b_fc2':tf.Variable(tf.random_normal([2048])),
          'out':tf.Variable(tf.random_normal([n_classes])),}

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def conv_neural_network(x):
    x=tf.reshape(x,shape=[-1,80,160,3])

    conv1=tf.nn.relu(conv2d(x,weights['W_conv1'])+biases['b_conv1'])
    conv1=maxpool2d(conv1)

    conv2=tf.nn.relu(conv2d(conv1,weights['W_conv2'])+biases['b_conv2'])
    conv2=maxpool2d(conv2)

    conv3=tf.nn.relu(conv2d(conv2,weights['W_conv3'])+biases['b_conv3'])
    conv3=maxpool2d(conv3)

    fc1=tf.reshape(conv3,[-1,10*20*128])
    fc1=tf.nn.relu(tf.matmul(fc1,weights['W_fc1'])+biases['b_fc1'])

    fc2=tf.nn.relu(tf.matmul(fc1,weights['W_fc2'])+biases['b_fc2'])
    fc2=tf.nn.dropout(fc2,dropout)

    output=tf.matmul(fc2,weights['out'])+biases['out']

    return output

def genTrainingSet(size):
    train_x=[]
    train_y=[]
    for i in range(size):
        image,code=gp.generateData()
        train_x.append(image)
        train_y.append(code)

    train_x=[image.flatten() for image in train_x]
    train_y=gf.formatPlates(train_y)

    return train_x,train_y

def train_neural_network(x):
    hm_epochs=10
    t_sets=10
    set_size=1000
    l_rate=0.0005

    with tf.name_scope("Model"):
        prediction = conv_neural_network(x)
    with tf.name_scope("Loss"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
    with tf.name_scope("AdamOpt"):
        optimizer = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(cost)

    prediction=tf.reshape(prediction,[-1,7,len(CHARS)])

    res=tf.argmax(prediction,2)
    act=tf.argmax(tf.reshape(y,[-1,7,len(CHARS)]), 2)

    correct = tf.equal(res, act)
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    init=tf.global_variables_initializer()
    with tf.Session() as sess: 
        sess.run(init)

        # Create a summary to monitor cost tensor
        tf.summary.scalar("loss", cost)
        # Create a summary to monitor accuracy tensor
        tf.summary.scalar("accuracy", accuracy)
        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        saver=tf.train.Saver()

        try:
            for s in range(t_sets):
                train_x,train_y=genTrainingSet(set_size)
                print("Set ", s+1,' generated out of ', t_sets)

                for epoch in range(hm_epochs):
                    epoch_loss = 0
                    i=0
                    while(i<len(train_x)):
                        start=i
                        end=i+batch_size

                        batch_x=np.array(train_x[start:end])
                        batch_y=np.array(train_y[start:end])
                        i+=batch_size

                        _, c, summary = sess.run([optimizer, cost, merged_summary_op],feed_dict={x: batch_x, y: batch_y})
                        summary_writer.add_summary(summary, epoch*set_size + i)
                        epoch_loss += c
                    saver.save(sess,'model')
                    print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)

                print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))

        except KeyboardInterrupt:
            print("Halting Training")

        finally:
            print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))

            res=res.eval({x:test_x})
            act=act.eval({y:test_y})
            
            resCh=[]
            actCh=[]
            for i,val in enumerate(res):
                resCh.append([gf.CHARS[x] for x in res[i]])
                actCh.append([gf.CHARS[x] for x in act[i]])

            fin=zip(actCh,resCh)
            for a,r in fin:
                print(a)
                print(r)
                print()

train_neural_network(x)