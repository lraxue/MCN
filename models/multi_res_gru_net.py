# -*- coding: utf-8 -*-
# @Time    : 17-11-20 下午4:35
# @Author  : Fei Xue
# @Email   : feixue@pku.edu.cn
# @File    : multi_res_gru_net.py
# @Software: PyCharm Community Edition


import numpy as np

# Theano
import theano
import theano.tensor as tensor

from models.net import Net, tensor5
from lib.layers import TensorProductLayer, ConvLayer, PoolLayer, Unpool3DLayer, \
    LeakyReLU, SoftmaxWithLoss3D, Conv3DLayer, InputLayer, FlattenLayer, \
    FCConv3DLayer, TanhLayer, SigmoidLayer, ComplementLayer, AddLayer, \
    EltwiseMultiplyLayer, get_trainable_params, CConv3DLayer, Pool3DLayer


class MultiResidualGRUNet(Net):
    def network_definition(self):
        self.x = tensor5()
        self.is_x_tensor4 = False

        img_w = self.img_w
        img_h = self.img_h

        n_gru_vox = [32, 32, 16, 8, 4]
        n_convfilter = [16, 32, 64, 128, 256, 512]
        n_deconvfilter = [2, 2, 8, 32, 128]
        input_shape = (self.batch_size, 3, img_w, img_h)

        x = InputLayer(input_shape)
        conv1a = ConvLayer(x, (n_convfilter[0], 7, 7))
        conv1b = ConvLayer(conv1a, (n_convfilter[0], 3, 3))
        pool1 = PoolLayer(conv1b, padding=(0, 0))  # H/2->64

        conv2a = ConvLayer(pool1, (n_convfilter[1], 3, 3))
        conv2b = ConvLayer(conv2a, (n_convfilter[1], 3, 3))
        conv2c = ConvLayer(pool1, (n_convfilter[1], 1, 1))
        pool2 = PoolLayer(conv2c)  # H/4->32

        conv3a = ConvLayer(pool2, (n_convfilter[2], 3, 3))
        conv3b = ConvLayer(conv3a, (n_convfilter[2], 3, 3))
        conv3c = ConvLayer(pool2, (n_convfilter[2], 1, 1))
        pool3 = PoolLayer(conv3c, padding=(0, 0))  # H/8->16

        conv4a = ConvLayer(pool3, (n_convfilter[3], 3, 3))
        conv4b = ConvLayer(conv4a, (n_convfilter[3], 3, 3))
        conv4c = ConvLayer(pool3, (n_convfilter[3], 1, 1))
        pool4 = PoolLayer(conv4c, padding=(0, 0))  # H/16->8

        conv5a = ConvLayer(pool4, (n_convfilter[4], 3, 3))
        conv5b = ConvLayer(conv5a, (n_convfilter[4], 3, 3))
        conv5c = ConvLayer(pool4, (n_convfilter[4], 1, 1))  # H/32->4
        pool5 = PoolLayer(conv5c, padding=(0, 0))

        conv6a = ConvLayer(pool5, (n_convfilter[5], 3, 3))
        conv6b = ConvLayer(conv6a, (n_convfilter[5], 3, 3))
        conv6c = ConvLayer(pool5, (n_convfilter[5], 1, 1))  # H/32->4

        def encoder(x):
            input_ = InputLayer(input_shape, x)
            conv1a_ = ConvLayer(input_, (n_convfilter[0], 7, 7), params=conv1a.params)
            rect1a_ = LeakyReLU(conv1a_)
            conv1b_ = ConvLayer(rect1a_, (n_convfilter[0], 3, 3), params=conv1b.params)
            rect1b_ = LeakyReLU(conv1b_)
            pool1_ = PoolLayer(rect1b_, padding=(0, 0))

            conv2a_ = ConvLayer(pool1_, (n_convfilter[1], 3, 3), params=conv2a.params)
            rect2a_ = LeakyReLU(conv2a_)
            conv2b_ = ConvLayer(rect2a_, (n_convfilter[1], 3, 3), params=conv2b.params)
            rect2b_ = LeakyReLU(conv2b_)
            conv2c_ = ConvLayer(pool1_, (n_convfilter[1], 1, 1), params=conv2c.params)
            res2_ = AddLayer(conv2c_, rect2b_)
            pool2_ = PoolLayer(res2_)

            conv3a_ = ConvLayer(pool2_, (n_convfilter[2], 3, 3), params=conv3a.params)
            rect3a_ = LeakyReLU(conv3a_)
            conv3b_ = ConvLayer(rect3a_, (n_convfilter[2], 3, 3), params=conv3b.params)
            rect3b_ = LeakyReLU(conv3b_)
            conv3c_ = ConvLayer(pool2_, (n_convfilter[2], 1, 1), params=conv3c.params)
            res3_ = AddLayer(conv3c_, rect3b_)
            pool3_ = PoolLayer(res3_, padding=(0, 0))

            conv4a_ = ConvLayer(pool3_, (n_convfilter[3], 3, 3), params=conv4a.params)
            rect4a_ = LeakyReLU(conv4a_)
            conv4b_ = ConvLayer(rect4a_, (n_convfilter[3], 3, 3), params=conv4b.params)
            rect4b_ = LeakyReLU(conv4b_)
            conv4c_ = ConvLayer(pool3_, (n_convfilter[3], 1, 1), params=conv4c.params)
            res4_ = AddLayer(conv4c_, rect4b_)
            pool4_ = PoolLayer(res4_, padding=(0, 0))

            conv5a_ = ConvLayer(pool4_, (n_convfilter[4], 3, 3), params=conv5a.params)
            rect5a_ = LeakyReLU(conv5a_)
            conv5b_ = ConvLayer(rect5a_, (n_convfilter[4], 3, 3), params=conv5b.params)
            rect5b_ = LeakyReLU(conv5b_)
            conv5c_ = ConvLayer(pool4_, (n_convfilter[4], 1, 1), params=conv5c.params)
            res5_ = AddLayer(conv5c_, rect5b_)
            pool5_ = PoolLayer(res5_, padding=(0, 0))

            conv6a_ = ConvLayer(pool5_, (n_convfilter[5], 3, 3), params=conv6a.params)
            rect6a_ = LeakyReLU(conv6a_)
            conv6b_ = ConvLayer(rect6a_, (n_convfilter[5], 3, 3), params=conv6b.params)
            rect6b_ = LeakyReLU(conv6b_)
            conv6c_ = ConvLayer(pool5_, (n_convfilter[5], 1, 1), params=conv6c.params)
            res6_ = AddLayer(conv6c_, rect6b_)

            flat3_ = FlattenLayer(res3_)
            flat4_ = FlattenLayer(res4_)
            flat5_ = FlattenLayer(res5_)
            flat6_ = FlattenLayer(res6_)

            print("res3: ", res3_.output_shape)
            print("res4: ", res4_.output_shape)
            print("res5: ", res5_.output_shape)
            print("res6: ", res6_.output_shape)
            # pool6_ = PoolLayer(res6_)

            return flat3_.output, flat4_.output, flat5_.output, flat6_.output
            # return flat6_.output

        # Set the shape of each resolution
        s_shape5 = (self.batch_size, n_gru_vox[4], n_deconvfilter[4], n_gru_vox[4], n_gru_vox[4])
        s_shape4 = (self.batch_size, n_gru_vox[3], n_deconvfilter[3], n_gru_vox[3], n_gru_vox[3])
        s_shape3 = (self.batch_size, n_gru_vox[2], n_deconvfilter[2], n_gru_vox[2], n_gru_vox[2])
        s_shape2 = (self.batch_size, n_gru_vox[1], n_deconvfilter[1], n_gru_vox[1], n_gru_vox[1])

        ## resolution 5
        prev_s5 = InputLayer(s_shape5)
        curr_s5 = InputLayer(s_shape5)
        t_x_s_update5 = CConv3DLayer(prev_s5, curr_s5, (n_deconvfilter[4], n_deconvfilter[4], 3, 3, 3))
        t_x_s_reset5 = CConv3DLayer(prev_s5, curr_s5, (n_deconvfilter[4], n_deconvfilter[4], 3, 3, 3))
        reset_gate5 = SigmoidLayer(t_x_s_reset5)
        rs5 = EltwiseMultiplyLayer(reset_gate5, prev_s5)
        t_x_rs5 = CConv3DLayer(rs5, curr_s5, (n_deconvfilter[4], n_deconvfilter[4], 3, 3, 3))

        ## resolution 4
        prev_s4 = InputLayer(s_shape4)
        curr_s4 = InputLayer(s_shape4)
        t_x_s_update4 = CConv3DLayer(prev_s4, curr_s4, (n_deconvfilter[3], n_deconvfilter[3], 3, 3, 3))
        t_x_s_reset4 = CConv3DLayer(prev_s4, curr_s4, (n_deconvfilter[3], n_deconvfilter[3], 3, 3, 3))
        reset_gate4 = SigmoidLayer(t_x_s_reset4)
        rs4 = EltwiseMultiplyLayer(reset_gate4, prev_s4)
        t_x_rs4 = CConv3DLayer(rs4, curr_s4, (n_deconvfilter[3], n_deconvfilter[3], 3, 3, 3))

        # resolution 3
        prev_s3 = InputLayer(s_shape3)
        curr_s3 = InputLayer(s_shape3)
        t_x_s_update3 = CConv3DLayer(prev_s3, curr_s3, (n_deconvfilter[2], n_deconvfilter[2], 3, 3, 3))
        t_x_s_reset3 = CConv3DLayer(prev_s3, curr_s3, (n_deconvfilter[2], n_deconvfilter[2], 3, 3, 3))
        reset_gate3 = SigmoidLayer(t_x_s_reset3)
        rs3 = EltwiseMultiplyLayer(reset_gate3, prev_s3)
        t_x_rs3 = CConv3DLayer(rs3, curr_s3, (n_deconvfilter[2], n_deconvfilter[2], 3, 3, 3))

        # resolution 4
        prev_s2 = InputLayer(s_shape2)
        curr_s2 = InputLayer(s_shape2)
        t_x_s_update2 = CConv3DLayer(prev_s2, curr_s2, (n_deconvfilter[1], n_deconvfilter[1], 3, 3, 3))
        t_x_s_reset2 = CConv3DLayer(prev_s2, curr_s2, (n_deconvfilter[1], n_deconvfilter[1], 3, 3, 3))
        reset_gate2 = SigmoidLayer(t_x_s_reset2)
        rs2 = EltwiseMultiplyLayer(reset_gate2, prev_s2)
        t_x_rs2 = CConv3DLayer(rs2, curr_s2, (n_deconvfilter[1], n_deconvfilter[1], 3, 3, 3))

        def gru5(curr_s5, prev_s5):
            curr_s5 = tensor.reshape(curr_s5, s_shape5)
            curr_s5_ = InputLayer(s_shape5, curr_s5)

            prev_s5_ = InputLayer(s_shape5, prev_s5)

            print("curr_s5: ", curr_s5_.output_shape)
            print("prev_s5: ", prev_s5_.output_shape)
            t_x_s_update5_ = CConv3DLayer(prev_s5_, curr_s5_, (n_deconvfilter[4], n_deconvfilter[4], 3, 3, 3),
                                          params=t_x_s_update5.params)
            t_x_s_reset5_ = CConv3DLayer(prev_s5_, curr_s5_, (n_deconvfilter[4], n_deconvfilter[4], 3, 3, 3),
                                         params=t_x_s_reset5.params)

            update5_ = SigmoidLayer(t_x_s_update5_)
            comp_udpate_gate5_ = ComplementLayer(update5_)
            reset_gate5_ = SigmoidLayer(t_x_s_reset5_)

            rs5_ = EltwiseMultiplyLayer(reset_gate5_, prev_s5_)
            t_x_rs5_ = CConv3DLayer(rs5_, curr_s5_, (n_deconvfilter[4], n_deconvfilter[4], 3, 3, 3),
                                    params=t_x_rs5.params)
            tanh_t_x_rs5_ = TanhLayer(t_x_rs5_)

            print("t_x_s_update5: ", t_x_s_update5_.output_shape)
            print("t_x_s_reset5: ", t_x_s_reset5_.output_shape)

            gru_out5_ = AddLayer(
                EltwiseMultiplyLayer(update5_, prev_s5_),
                EltwiseMultiplyLayer(comp_udpate_gate5_, tanh_t_x_rs5_))

            print("gru_out5: ", gru_out5_.output_shape)
            return gru_out5_.output

        def gru4(curr_s4, prev_s4):
            curr_s4 = tensor.reshape(curr_s4, s_shape4)
            curr_s4_ = InputLayer(s_shape4, curr_s4)
            prev_s4_ = InputLayer(s_shape4, prev_s4)

            t_x_s_update4_ = CConv3DLayer(prev_s4_, curr_s4_, (n_deconvfilter[3], n_deconvfilter[3], 3, 3, 3),
                                          params=t_x_s_update4.params)
            t_x_s_reset4_ = CConv3DLayer(prev_s4_, curr_s4_, (n_deconvfilter[3], n_deconvfilter[3], 3, 3, 3),
                                         params=t_x_s_reset4.params)

            update4_ = SigmoidLayer(t_x_s_update4_)
            comp_udpate_gate4_ = ComplementLayer(update4_)
            reset_gate4_ = SigmoidLayer(t_x_s_reset4_)

            rs4_ = EltwiseMultiplyLayer(reset_gate4_, prev_s4_)
            t_x_rs4_ = CConv3DLayer(rs4_, curr_s4_, (n_deconvfilter[3], n_deconvfilter[3], 3, 3, 3),
                                    params=t_x_rs4.params)
            tanh_t_x_rs4_ = TanhLayer(t_x_rs4_)

            gru_out4_ = AddLayer(
                EltwiseMultiplyLayer(update4_, prev_s4_),
                EltwiseMultiplyLayer(comp_udpate_gate4_, tanh_t_x_rs4_))

            print("gru_out4: ", gru_out4_.output_shape)

            return gru_out4_.output

        def gru3(curr_s3, prev_s3):
            curr_s3 = tensor.reshape(curr_s3, s_shape3)
            curr_s3_ = InputLayer(s_shape3, curr_s3)
            prev_s3_ = InputLayer(s_shape3, prev_s3)

            t_x_s_update3_ = CConv3DLayer(prev_s3_, curr_s3_, (n_deconvfilter[2], n_deconvfilter[2], 3, 3, 3),
                                          params=t_x_s_update3.params)
            t_x_s_reset3_ = CConv3DLayer(prev_s3_, curr_s3_, (n_deconvfilter[2], n_deconvfilter[2], 3, 3, 3),
                                         params=t_x_s_reset3.params)

            update3_ = SigmoidLayer(t_x_s_update3_)
            comp_udpate_gate3_ = ComplementLayer(update3_)
            reset_gate3_ = SigmoidLayer(t_x_s_reset3_)

            rs3_ = EltwiseMultiplyLayer(reset_gate3_, prev_s3_)
            t_x_rs3_ = CConv3DLayer(rs3_, curr_s3_, (n_deconvfilter[2], n_deconvfilter[2], 3, 3, 3),
                                    params=t_x_rs3.params)
            tanh_t_x_rs3_ = TanhLayer(t_x_rs3_)

            gru_out3_ = AddLayer(
                EltwiseMultiplyLayer(update3_, prev_s3_),
                EltwiseMultiplyLayer(comp_udpate_gate3_, tanh_t_x_rs3_))

            print("gru_out3: ", gru_out3_.output_shape)

            return gru_out3_.output

        def gru2(curr_s2, prev_s2):
            curr_s2 = tensor.reshape(curr_s2, s_shape2)
            curr_s2_ = InputLayer(s_shape2, curr_s2)
            prev_s2_ = InputLayer(s_shape2, prev_s2)

            t_x_s_update2_ = CConv3DLayer(prev_s2_, curr_s2_, (n_deconvfilter[1], n_deconvfilter[1], 3, 3, 3),
                                          params=t_x_s_update2.params)
            t_x_s_reset2_ = CConv3DLayer(prev_s2_, curr_s2_, (n_deconvfilter[1], n_deconvfilter[1], 3, 3, 3),
                                         params=t_x_s_reset2.params)

            update2_ = SigmoidLayer(t_x_s_update2_)
            comp_udpate_gate2_ = ComplementLayer(update2_)
            reset_gate2_ = SigmoidLayer(t_x_s_reset2_)

            rs2_ = EltwiseMultiplyLayer(reset_gate2_, prev_s2_)
            t_x_rs2_ = CConv3DLayer(rs2_, curr_s2_, (n_deconvfilter[1], n_deconvfilter[1], 3, 3, 3),
                                    params=t_x_rs2.params)
            tanh_t_x_rs2_ = TanhLayer(t_x_rs2_)

            gru_out2_ = AddLayer(
                EltwiseMultiplyLayer(update2_, prev_s2_),
                EltwiseMultiplyLayer(comp_udpate_gate2_, tanh_t_x_rs2_))

            print("gru_out2: ", gru_out2_.output_shape)
            return gru_out2_.output

        s_encoder, _ = theano.scan(encoder,
                                sequences=[self.x])

        # print("self.x: ", self.x)
        out_encoder5 = s_encoder[3]
        out_encoder4 = s_encoder[2]
        out_encoder3 = s_encoder[1]
        out_encoder2 = s_encoder[0]

        s_gru5, _ = theano.scan(gru5,
                                sequences=[out_encoder5],
                                outputs_info=[tensor.zeros_like(np.zeros(s_shape5),
                                                                dtype=theano.config.floatX)])

        input_5 = InputLayer(s_shape5, s_gru5[-1])
        # print("input_5: ", input_5.output_shape)
        pred5 = Conv3DLayer(input_5, (2, 3, 3, 3))
        unpool5 = Unpool3DLayer(input_5)
        conv3d5 = Conv3DLayer(unpool5, (n_deconvfilter[3], 3, 3, 3))
        rect3d5 = LeakyReLU(conv3d5)

        print("rect3d5: ", rect3d5.output_shape)

        print("recct3d5: ", rect3d5.output)

        s_gru4, _ = theano.scan(gru4,
                                sequences=[out_encoder4],
                                outputs_info=[rect3d5.output]
                                )

        input_4 = InputLayer(s_shape4, s_gru4[-1])
        pred4 = Conv3DLayer(input_4, (2, 3, 3, 3))

        unpool4 = Unpool3DLayer(input_4)
        conv3d4 = Conv3DLayer(unpool4, (n_deconvfilter[2], 3, 3, 3))
        rect3d4 = LeakyReLU(conv3d4)
        print("rect3d4: ", rect3d4.output_shape)
        print("recct3d4: ", rect3d4.output)

        s_gru3, _ = theano.scan(gru3,
                                sequences=[out_encoder3],
                                outputs_info=[rect3d4.output])

        input_3 = InputLayer(s_shape3, s_gru3[-1])
        pred3 = Conv3DLayer(input_3, (2, 3, 3, 3))

        unpool3 = Unpool3DLayer(input_3)
        conv3d3 = Conv3DLayer(unpool3, (n_deconvfilter[1], 3, 3, 3))
        rect3d3 = LeakyReLU(conv3d3)

        print("rect3d3: ", rect3d3.output_shape)
        print("recct3d3: ", rect3d3.output)
        s_gru2, _ = theano.scan(gru2,
                                sequences=[out_encoder2],
                                outputs_info=[rect3d3.output])

        input_2 = InputLayer(s_shape2, s_gru2[-1])
        pred2 = Conv3DLayer(input_2, (2, 3, 3, 3))

        labele_shape = self.y.shape
        label3 = self.y[:, 0:labele_shape[1]:2, :, 0:labele_shape[3]:2, 0:labele_shape[4]:2]
        label4 = self.y[:, 0:labele_shape[1]:4, :, 0:labele_shape[3]:4, 0:labele_shape[4]:4]
        label5 = self.y[:, 0:labele_shape[1]:8, :, 0:labele_shape[3]:8, 0:labele_shape[4]:8]


        # print("pred5: ", pred5.output_shape)
        # print("pred4: ", pred4.output_shape)
        # print("pred3: ", pred3.output_shape)
        print("pred2: ", pred2.output_shape)

        softmax_loss5 = SoftmaxWithLoss3D(pred5.output)
        softmax_loss4 = SoftmaxWithLoss3D(pred4.output)
        softmax_loss3 = SoftmaxWithLoss3D(pred3.output)
        softmax_loss2 = SoftmaxWithLoss3D(pred2.output)

        # self.loss = softmax_loss2.loss(self.y)
        self.loss = (softmax_loss5.loss(label5)  + softmax_loss4.loss(label4) +
                     softmax_loss3.loss(label3) + softmax_loss2.loss(self.y)) / 4.
        # self.loss = self.loss /
        self.error = softmax_loss2.error(self.y)
        self.output = softmax_loss2.prediction()
        self.params = get_trainable_params()
        self.activations = []
