# Pix2Vox-ResNet

This project is part of the course "Machine Learning for 3D Geometry" from Technical University of Munich, offered in Summer Semester 2022.

![Overview](./Pix2Vox-Overview.jpg)
(source: https://github.com/hzxie/Pix2Vox)

This project is largely based on previous work from Pix2Vox (Xie et al. 2019).
Pix2Vox is an encoder-decoder-based framework for single-view or multi-view 3D shape reconstruction. 
It utilizes a pre-trained VGG-16 as part of its encoder to extract visual representations from the input images. 
In our project, we conduct an ablation study on Pix2Vox’s encoder by replacing the pre-trained VGG-16 with a pre-trained ResNet-152, which has been shown to gain higher accuracy on the ImageNet benchmark. 
Our aim is to verify whether learned 2D features from pre-trained models with better accuracy on ImageNet can be used to generate 3D shape reconstructions of better quality. Experimental results on the ShapeNet benchmark indicate that the model using ResNet-152 gains an absolute IoU increase of 0.3% when using more than 12 views, while underperforming the model using VGG-16 when a small amount of views is used.

# Teams

- Cuong Nguyen
- Michelle Espranita Liman
- Thang Tran
- Viet Nguyen

# Test Results

## 1 view

```
============================ TEST RESULTS ============================
Taxonomy        #Sample Baseline        t=0.20  t=0.30  t=0.40  t=0.50
aeroplane       810     0.5130          0.6047  0.6235  0.6314  0.6289
bench           364     0.4210          0.5240  0.5406  0.5485  0.5481
cabinet         315     0.7160          0.7609  0.7623  0.7600  0.7517
car             1501    0.7980          0.8206  0.8327  0.8389  0.8402
chair           1357    0.4660          0.5249  0.5320  0.5326  0.5260
display         220     0.4680          0.4973  0.5019  0.5007  0.4926
lamp            465     0.3810          0.4335  0.4318  0.4251  0.4123
speaker         325     0.6620          0.6920  0.6905  0.6850  0.6742
rifle           475     0.5440          0.5485  0.5711  0.5790  0.5746
sofa            635     0.6280          0.6772  0.6835  0.6844  0.6787
table           1703    0.5130          0.5655  0.5749  0.5791  0.5776
telephone       211     0.6610          0.7302  0.7395  0.7458  0.7502
watercraft      389     0.5130          0.5489  0.5627  0.5675  0.5633
Overall                                 0.6182  0.6281  0.6313  0.6277
```

## 2 views

```
============================ TEST RESULTS ============================
Taxonomy        #Sample Baseline        t=0.20  t=0.30  t=0.40  t=0.50
aeroplane       810     0.5360          0.6438  0.6638  0.6693  0.6629
bench           364     0.4840          0.5787  0.5960  0.6019  0.5969
cabinet         315     0.7460          0.7916  0.7953  0.7942  0.7885
car             1501    0.8210          0.8438  0.8565  0.8626  0.8636
chair           1357    0.5150          0.5791  0.5869  0.5857  0.5763
display         220     0.5270          0.5539  0.5582  0.5567  0.5470
lamp            465     0.4060          0.4617  0.4583  0.4473  0.4291
speaker         325     0.6960          0.7256  0.7263  0.7229  0.7137
rifle           475     0.5820          0.6031  0.6244  0.6300  0.6217
sofa            635     0.6770          0.7220  0.7308  0.7328  0.7283
table           1703    0.5500          0.5990  0.6090  0.6121  0.6090
telephone       211     0.7170          0.7731  0.7830  0.7881  0.7907
watercraft      389     0.5760          0.6034  0.6186  0.6211  0.6138
Overall                                 0.6578  0.6685  0.6706  0.6651
```

## 3 views

```
============================ TEST RESULTS ============================
Taxonomy        #Sample Baseline        t=0.20  t=0.30  t=0.40  t=0.50
aeroplane       810     0.5490          0.6602  0.6806  0.6854  0.6770
bench           364     0.5020          0.6079  0.6242  0.6276  0.6212
cabinet         315     0.7630          0.8014  0.8050  0.8040  0.7986
car             1501    0.8290          0.8508  0.8637  0.8698  0.8708
chair           1357    0.5330          0.5979  0.6053  0.6032  0.5934
display         220     0.5450          0.5741  0.5799  0.5798  0.5717
lamp            465     0.4150          0.4715  0.4678  0.4554  0.4347
speaker         325     0.7080          0.7387  0.7390  0.7348  0.7260
rifle           475     0.5930          0.6234  0.6456  0.6488  0.6382
sofa            635     0.6900          0.7371  0.7444  0.7449  0.7398
table           1703    0.5640          0.6143  0.6239  0.6262  0.6224
telephone       211     0.7320          0.7950  0.8044  0.8093  0.8120
watercraft      389     0.5960          0.6182  0.6322  0.6338  0.6248
Overall                                 0.6729  0.6833  0.6846  0.6784
```

## 4 views

```
============================ TEST RESULTS ============================
Taxonomy        #Sample Baseline        t=0.20  t=0.30  t=0.40  t=0.50
aeroplane       810     0.5560          0.6684  0.6882  0.6915  0.6822
bench           364     0.5160          0.6185  0.6347  0.6388  0.6305
cabinet         315     0.7670          0.8057  0.8095  0.8087  0.8033
car             1501    0.8330          0.8548  0.8675  0.8733  0.8739
chair           1357    0.5410          0.6069  0.6138  0.6109  0.6004
display         220     0.5580          0.5852  0.5892  0.5863  0.5776
lamp            465     0.4160          0.4763  0.4705  0.4570  0.4354
speaker         325     0.7140          0.7416  0.7422  0.7385  0.7299
rifle           475     0.5950          0.6349  0.6582  0.6610  0.6481
sofa            635     0.6980          0.7441  0.7523  0.7534  0.7483
table           1703    0.5730          0.6236  0.6323  0.6333  0.6281
telephone       211     0.7380          0.7995  0.8108  0.8164  0.8188
watercraft      389     0.6040          0.6251  0.6384  0.6385  0.6291
Overall                                 0.6803  0.6904  0.6910  0.6840
```

## 5 views

```
============================ TEST RESULTS ============================
Taxonomy        #Sample Baseline        t=0.20  t=0.30  t=0.40  t=0.50
aeroplane       810     0.5610          0.6735  0.6936  0.6963  0.6860
bench           364     0.5270          0.6261  0.6419  0.6442  0.6356
cabinet         315     0.7720          0.8074  0.8108  0.8090  0.8031
car             1501    0.8360          0.8567  0.8694  0.8752  0.8758
chair           1357    0.5500          0.6126  0.6194  0.6164  0.6055
display         220     0.5650          0.5923  0.5960  0.5913  0.5806
lamp            465     0.4210          0.4787  0.4724  0.4577  0.4358
speaker         325     0.7170          0.7455  0.7469  0.7438  0.7363
rifle           475     0.6000          0.6383  0.6602  0.6624  0.6485
sofa            635     0.7060          0.7493  0.7573  0.7584  0.7535
table           1703    0.5800          0.6302  0.6383  0.6386  0.6327
telephone       211     0.7540          0.8079  0.8183  0.8230  0.8243
watercraft      389     0.6100          0.6303  0.6437  0.6441  0.6340
Overall                                 0.6851  0.6950  0.6951  0.6876
```

## 8 views

```
============================ TEST RESULTS ============================
Taxonomy        #Sample Baseline        t=0.20  t=0.30  t=0.40  t=0.50
aeroplane       810     N/a             0.6802  0.6999  0.7024  0.6914
bench           364     N/a             0.6385  0.6540  0.6545  0.6434
cabinet         315     N/a             0.8131  0.8175  0.8169  0.8126
car             1501    N/a             0.8591  0.8721  0.8779  0.8787
chair           1357    N/a             0.6217  0.6279  0.6242  0.6127
display         220     N/a             0.6060  0.6104  0.6063  0.5948
lamp            465     N/a             0.4850  0.4787  0.4628  0.4393
speaker         325     N/a             0.7510  0.7524  0.7499  0.7425
rifle           475     N/a             0.6502  0.6702  0.6708  0.6569
sofa            635     N/a             0.7593  0.7671  0.7673  0.7617
table           1703    N/a             0.6372  0.6442  0.6437  0.6376
telephone       211     N/a             0.8212  0.8312  0.8363  0.8375
watercraft      389     N/a             0.6378  0.6513  0.6513  0.6403
Overall                                 0.6925  0.7020  0.7016  0.6938
```

## 12 views

```
============================ TEST RESULTS ============================
Taxonomy        #Sample Baseline        t=0.20  t=0.30  t=0.40  t=0.50
aeroplane       810     N/a             0.6835  0.7033  0.7056  0.6938
bench           364     N/a             0.6422  0.6568  0.6561  0.6441
cabinet         315     N/a             0.8141  0.8179  0.8175  0.8126
car             1501    N/a             0.8610  0.8740  0.8798  0.8806
chair           1357    N/a             0.6276  0.6335  0.6293  0.6173
display         220     N/a             0.6111  0.6138  0.6095  0.5990
lamp            465     N/a             0.4881  0.4818  0.4643  0.4398
speaker         325     N/a             0.7524  0.7538  0.7509  0.7437
rifle           475     N/a             0.6554  0.6743  0.6752  0.6596
sofa            635     N/a             0.7645  0.7721  0.7720  0.7659
table           1703    N/a             0.6420  0.6487  0.6480  0.6418
telephone       211     N/a             0.8191  0.8280  0.8325  0.8339
watercraft      389     N/a             0.6419  0.6559  0.6566  0.6456
Overall                                 0.6963  0.7055  0.7049  0.6967
```

## 16 views

```
============================ TEST RESULTS ============================
Taxonomy        #Sample Baseline        t=0.20  t=0.30  t=0.40  t=0.50
aeroplane       810     N/a             0.6853  0.7050  0.7077  0.6952
bench           364     N/a             0.6457  0.6594  0.6588  0.6481
cabinet         315     N/a             0.8156  0.8190  0.8182  0.8137
car             1501    N/a             0.8620  0.8749  0.8806  0.8812
chair           1357    N/a             0.6306  0.6365  0.6320  0.6200
display         220     N/a             0.6157  0.6194  0.6157  0.6046
lamp            465     N/a             0.4891  0.4817  0.4634  0.4393
speaker         325     N/a             0.7552  0.7567  0.7536  0.7463
rifle           475     N/a             0.6579  0.6766  0.6772  0.6617
sofa            635     N/a             0.7668  0.7745  0.7752  0.7701
table           1703    N/a             0.6449  0.6516  0.6507  0.6446
telephone       211     N/a             0.8294  0.8390  0.8426  0.8424
watercraft      389     N/a             0.6455  0.6586  0.6593  0.6485
Overall                                 0.6988  0.7079  0.7072  0.6990
```

## 20 views

```
============================ TEST RESULTS ============================
Taxonomy        #Sample Baseline        t=0.20  t=0.30  t=0.40  t=0.50
aeroplane       810     N/a             0.6868  0.7064  0.7085  0.6964
bench           364     N/a             0.6468  0.6591  0.6579  0.6463
cabinet         315     N/a             0.8154  0.8188  0.8181  0.8140
car             1501    N/a             0.8627  0.8756  0.8814  0.8819
chair           1357    N/a             0.6325  0.6384  0.6339  0.6215
display         220     N/a             0.6148  0.6169  0.6109  0.5995
lamp            465     N/a             0.4894  0.4817  0.4639  0.4389
speaker         325     N/a             0.7550  0.7560  0.7530  0.7459
rifle           475     N/a             0.6600  0.6776  0.6778  0.6632
sofa            635     N/a             0.7683  0.7757  0.7763  0.7714
table           1703    N/a             0.6462  0.6531  0.6522  0.6462
telephone       211     N/a             0.8331  0.8416  0.8443  0.8439
watercraft      389     N/a             0.6477  0.6607  0.6601  0.6485
Overall                                 0.7001  0.7089  0.7081  0.6998
```