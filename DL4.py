#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


mnist = tf.keras.datasets.mnist


# In[ ]:


#Nikolas Frankson


# In[3]:


learning_rate = 0.0001


# In[4]:


batch_size = 100


# In[5]:


update_step = 10


# In[ ]:


#Nikolas Frankson


# In[6]:


layer_1_nodes = 500


# In[7]:


layer_2_nodes = 500


# In[8]:


layer_3_nodes = 500


# In[9]:


output_nodes = 10


# In[ ]:


#Nikolas Frankson


# In[10]:


network_input = tf.placeholder(tf.float32, [None,784])


# In[11]:


target_output = tf.placeholder(tf.float32,[None,output_nodes])


# In[ ]:


#Nikolas Frankson


# In[12]:


layer_1 = tf.Variable(tf.random_normal([784, layer_1_nodes]))


# In[13]:


layer_1_bias = tf.Variable(tf.random_normal([layer_1_nodes]))


# In[14]:


layer_2 = tf.Variable(tf.random_normal([layer_1_nodes, layer_2_nodes]))


# In[15]:


layer_2_bias = tf.Variable(tf.random_normal([layer_2_nodes]))


# In[16]:


layer_3 = tf.Variable(tf.random_normal([layer_2_nodes,layer_3_nodes]))


# In[22]:


layer_3_bias = tf.Variable(tf.random_normal([layer_3_nodes]))


# In[18]:


out_layer = tf.Variable(tf.random_normal([layer_3_nodes, output_nodes]))


# In[19]:


out_layer_bias = tf.Variable(tf.random_normal([output_nodes]))


# In[20]:


l1_output = tf.nn.relu(tf.matmul(network_input, layer_1)+ layer_1_bias)


# In[21]:


l2_output = tf.nn.relu(tf.matmul(l1_output, layer_2)+ layer_2_bias)


# In[23]:


l3_output = tf.nn.relu(tf.matmul(l2_output, layer_3)+ layer_3_bias)


# In[24]:


ntwk_output_1 = tf.matmul(l3_output, out_layer) + out_layer_bias


# In[25]:


ntwk_output_2 = tf.nn.softmax(ntwk_output_1)


# In[ ]:


#Nikolas Frankson


# In[26]:


cf = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=ntwk_output_1, labels=target_output))


# In[27]:


ts = tf.train.GradientDescentOptimizer(learning_rate).minimize(cf)


# In[28]:


cp = tf.equal(tf.argmax(ntwk_output_2, 1), tf.argmax(target_output, 1))


# In[29]:


acc = tf.reduce_mean(tf.cast(cp, tf.float32))


# In[ ]:


#Nikolas Frankson


# In[39]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_epochs = 10
    for epoch in range(num_epochs):
        total_cost = 0
        for_in range(int(mnist.train.num_examples / batch_size)):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
        t, c = sess.run([ts,cf], feed_dict={network_input: batch_x, target_output: batch_y})
        total_cost += c
        print('Epoch', epoch, 'completed out of', num_epochs, 'loss:', total_cost)
    print('Accuracy:', acc.eval({network_input: mnist.test.images, target_output: mnist.test.labels}))


# In[ ]:


#Nikolas Frankson

