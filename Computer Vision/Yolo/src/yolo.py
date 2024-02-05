import torch
import torch.nn as nn 
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    # first route from the end of the previous block
    (512, 3, 2),
    ["B", 8], 
    # second route from the end of the previous block
    (1024, 3, 2),
    ["B", 4],
    # until here is YOLO-53
    (512, 1, 1),
    (1024, 3, 1),
    (512, 1, 1),
    (1024, 3, 1),
    (512, 1, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    (256, 1, 1),
    (512, 3, 1),
    (256, 1, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    (128, 1, 1),
    (256, 3, 1),
    (128, 1, 1),
    "S",
]
class CONV(nn.Module):
    def __init__(self, in_channels , out_channels ,batch_normalization_and_activatuion:bool=True,**kwargs):
        super().__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,bias=not batch_normalization_and_activatuion,**kwargs)
        self.bn=nn.BatchNorm2d(out_channels)
        self.leaky =nn.LeakyReLU(0.1)
        self.use_bn_and_activation=batch_normalization_and_activatuion
    def forward(self,x):
        if self.use_bn_and_activation:
            return self.leaky(self.bn(self.conv(x)))
        else :
            return self.conv(x)

class Residual(nn.Module):
    def __init__(self,channels,num_repeats, **kwargs):
        super().__init__()
        self.layers=nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers+=[
                nn.Sequential(
                CONV(channels,channels//2,kernel_size=1),
                CONV(channels//2,channels,kernel_size=3,padding=1)
                )
            ]
        self.num_repeats=num_repeats

    def forward(self,x):
        for layer in self.layers:
            x=layer(x)+x
        

        return x 

class ScalePredection(nn.Module):
    def __init__(self,in_channel, num_classes) :
        super().__init__()
        self.head= nn.Sequential(
            CONV(in_channel,2*in_channel,kernel_size=3,padding=1),
            CONV(2*in_channel,(num_classes+5)*3,batch_normalization_and_activatuion=False,kernel_size=1)#[num_classes,po,x,y,w,h]x 3 anchors
        )
        self.num_classes=num_classes

    def forward(self,x):
        return self.head(x).reshape(x.shape[0],3,self.num_classes+5,x.shape[2],x.shape[3]).permute(0,1,3,4,2)
                # Batch*anchors*(classes+x,y,w,h,po)*x*y >> Batch*anchors*x*y*(classes+x,y,w,h,po)
class YOLOv3(nn.Module):
    def __init__(self, in_channels=3,num_classes=80) :
        super().__init__()
        self.num_classes=num_classes
        self.in_channels=in_channels
        self.layers=self.create_model()

    def forward(self,x):
        outputs = [] 
        route_connections = [] 
        for layer in self.layers:
            if isinstance(layer,ScalePredection):
                 outputs.append(layer(x))
                 continue
            x=layer(x)

            if isinstance(layer ,Residual) and layer.num_repeats==8: # because these are the layers that have route connections
                route_connections.append(x)

            elif isinstance(layer,nn.Upsample):
                x=torch.cat([x,route_connections[-1]],dim=1)
                route_connections.pop()
        return outputs
    


    def create_model(self,config_file_path="None"):
        layers=nn.ModuleList()
        in_channels=self.in_channels
        for module in config:
            if isinstance(module,tuple):
                out_channels,kernel_size,stride=module
                layers.append(CONV(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=1 if kernel_size==3 else 0))
                in_channels=out_channels
            if isinstance(module,list):
                num_reapeats=module[1]
                layers.append(Residual(in_channels,num_repeats=num_reapeats))
            elif isinstance(module,str):
                if module=="S":
                    layers.append(ScalePredection(in_channel=in_channels,num_classes=self.num_classes)) 
                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels=in_channels*3 # because whenever we upsample we will concatinate from skip connctions from previous layers 
                    #and typically in yolo v3 we concatenate double the size of channels of the current output 
        return layers
                
if __name__=="__main__":
    num_classes=80
    image_size=416
    model=YOLOv3(num_classes=num_classes)
    x=torch.randn((2,3,image_size,image_size))
    out=model(x)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
    print(model)