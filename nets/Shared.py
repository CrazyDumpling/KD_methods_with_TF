import tensorflow as tf
from nets import SVP

tcf = tf.contrib.framework
tcl = tf.contrib.layers

def FSP(students, teachers, weight = 1e-3):
    '''
    Junho Yim, Donggyu Joo, Jihoon Bae, and Junmo Kim.
    A gift from knowledge distillation: Fast optimization, network minimization and transfer learning. 
    In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 4133–4141, 2017.
    '''
    def Grammian(top, bot):
        with tf.variable_scope('Grammian'):
            t_sz = top.get_shape().as_list()
            b_sz = bot.get_shape().as_list()
    
            if t_sz[1] > b_sz[1]:
                top = tf.contrib.layers.max_pool2d(top, [2, 2], 2)
                            
            top = tf.reshape(top,[-1, b_sz[1]*b_sz[2], t_sz[-1]])
            bot = tf.reshape(bot,[-1, b_sz[1]*b_sz[2], b_sz[-1]])
    
            Gram = tf.matmul(top, bot, transpose_a = True)/(b_sz[1]*b_sz[2])
            return Gram
    with tf.variable_scope('FSP'):
        Dist_loss = []
        for i in range(len(students)-1):
            gs0 = Grammian(students[i], students[i+1])
            gt0 = Grammian(teachers[i], teachers[i+1])
     
            Dist_loss.append(tf.reduce_mean(tf.reduce_sum(tf.square(tf.stop_gradient(gt0)-gs0),[1,2])/2 ))

        return tf.add_n(Dist_loss)*weight
    
def KD_SVD(student_feature_maps, teacher_feature_maps, dist_type = 'SVD'):
    '''
    Seung Hyun Lee, Dae Ha Kim, and Byung Cheol Song.
    Self-supervised knowledge distillation using singular value decomposition. In
    European Conference on ComputerVision, pages 339–354. Springer, 2018.
    '''
    with tf.variable_scope('Distillation'):
        GNN_losses = []
        K = 1
        V_Tb = V_Sb = None
        for i, sfm, tfm in zip(range(len(student_feature_maps)), student_feature_maps, teacher_feature_maps):
            with tf.variable_scope('Compress_feature_map%d'%i):
                if dist_type == 'SVD':
                    Sigma_T, U_T, V_T = SVP.SVD(tfm, K, name = 'TSVD%d'%i)
                    Sigma_S, U_S, V_S = SVP.SVD(sfm, K+3, name = 'SSVD%d'%i)
                    B, D,_ = V_S.get_shape().as_list()
                    V_S, V_T = SVP.Align_rsv(V_S, V_T)
                    
                elif dist_type == 'EID':
                    Sigma_T, U_T, V_T = SVP.SVD_eid(tfm, K, name = 'TSVD%d'%i)
                    Sigma_S, U_S, V_S = SVP.SVD_eid(sfm, K+3, name = 'SSVD%d'%i)
                    B, D,_ = V_S.get_shape().as_list()
                    V_S, V_T = SVP.Align_rsv(V_S, V_T)
                
                Sigma_T = tf.expand_dims(Sigma_T,1)
                V_T *= Sigma_T
                V_S *= Sigma_T
                
            if i > 0:
                with tf.variable_scope('RBF%d'%i):    
                    S_rbf = tf.exp(-tf.square(tf.expand_dims(V_S,2)-tf.expand_dims(V_Sb,1))/8)
                    T_rbf = tf.exp(-tf.square(tf.expand_dims(V_T,2)-tf.expand_dims(V_Tb,1))/8)
                    
                    # tf.logging.info(tf.expand_dims(V_S,2).get_shape())
                    # tf.logging.info(tf.expand_dims(V_Sb,1).get_shape())
                    # tf.logging.info((tf.expand_dims(V_S,2)-tf.expand_dims(V_Sb,1)).get_shape())

                    l2loss = (S_rbf-tf.stop_gradient(T_rbf))**2
                    l2loss = tf.where(tf.is_finite(l2loss), l2loss, tf.zeros_like(l2loss))
                    # tf.logging.info(l2loss.get_shape())
                    GNN_losses.append(tf.reduce_sum(l2loss))
                    # tf.logging.info("next line")
            V_Tb = V_T
            V_Sb = V_S

        transfer_loss =  tf.add_n(GNN_losses)
        # tf.logging.info(transfer_loss.get_shape())

        return transfer_loss


def SVD_PLUS(student_feature_maps, teacher_feature_maps):
    '''
    ZiJie Song.
    '''
    with tf.variable_scope('SVD_PLUS'):
        GNN_losses = []
        K = 1
        size = [64,128,256]
        for i, sfm, tfm in zip(range(len(student_feature_maps)), student_feature_maps, teacher_feature_maps):
            with tf.variable_scope('Compress_feature_map%d'%i):
                Sigma_T, U_T, V_T = SVP.SVD(tfm, K, name = 'TSVD_%d'%i)
                Sigma_S, U_S, V_S = SVP.SVD(sfm, K+1, name = 'SSVD_%d'%i)
                B, D,_ = V_S.get_shape().as_list()
                V_S, V_T = SVP.Align_rsv(V_S, V_T)
                
                Sigma_T = tf.expand_dims(Sigma_T,1)
                V_T *= Sigma_T
                V_S *= Sigma_T

            with tcf.arg_scope([tcl.fully_connected, tcl.batch_norm], trainable = True):
                with tcf.arg_scope([tcl.batch_norm], is_training = True):
                    with tf.variable_scope('RBF%d'%i):
                        std = tcl.fully_connected(V_S , 1,
                                             biases_initializer = tf.zeros_initializer(),
                                             biases_regularizer = tcl.l2_regularizer(5e-4),
                                             scope = 'full_distill%d'%i)

                        # tf.logging.info(std.get_shape())
                        # tf.logging.info(V_T.get_shape())
                        # std = tf.zeros([1,1,size[i]])
                        std = tf.norm(std)
                        V_T = tf.norm(V_T)
                        l2loss = tf.square((std-tf.stop_gradient(V_T))**2)

                        # l2loss = tf.norm(l2loss)

                        l2loss = tf.where(tf.is_finite(l2loss), l2loss, tf.zeros_like(l2loss))                        

                        # tf.logging.info(l2loss.get_shape())
                        
                        # l2loss = tf.sigmoid(l2loss)
                        
                        GNN_losses.append(tf.reduce_sum(l2loss))

        transfer_loss = tf.add_n(GNN_losses)
        # tf.logging.info(transfer_loss.get_shape())

        return transfer_loss
