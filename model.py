import torch
from torch import nn

######################################
####       Helper functions     ######
######################################
def initialize_weights(layer, mean=0.0, std=0.02):
    if isinstance(layer, nn.Conv3d) or isinstance(layer, nn.ConvTranspose2d):
        torch.nn.init.normal_(layer.weight, mean, std)
    elif isinstance(layer, nn.Linear) or isinstance(layer, nn.BatchNorm2d):
        torch.nn.init.normal_(layer.weight, mean, std)
        torch.nn.init.constant_(layer.bias, 0)
class Reshape(nn.Module):
    def __init__(self, dims=[32, 1, 1]):
        super().__init__()
        self.dims=dims
        
    def forward(self, x):
        N = x.size(0)
        return x.view(N,*self.dims)
    
######################################
####     Temporal Network       ######
######################################
class TempNetwork(nn.Module):
    def __init__(self, z_dim=32, n_bars=2, hid_channels=1024):
        super().__init__()
        self.Input = Reshape(dims=[z_dim, 1, 1])
        
        self.Base = nn.Sequential()
        self.Base.add_module('upconv0', nn.ConvTranspose2d(z_dim, hid_channels,
                                        kernel_size=(2, 1), stride=(1, 1), padding=0))
        self.Base.add_module('bn0', nn.BatchNorm2d(hid_channels))
        self.Base.add_module('relu0', nn.ReLU())
        
        self.Base.add_module('upconv1', nn.ConvTranspose2d(hid_channels, z_dim,
                                        kernel_size=(n_bars-1, 1), stride=(1, 1), padding=0))
        self.Base.add_module('bn1', nn.BatchNorm2d(z_dim))
        self.Base.add_module('relu1', nn.ReLU())
        
        self.Output = Reshape(dims=[z_dim, n_bars])
        
    def forward(self, x):
        x = self.Input(x)
        x = self.Base(x)
        x = self.Output(x)
        return x

######################################
######     BarGenerator       ########
######################################
class BarGenerator(nn.Module):
    def __init__(self, z_dim=32, n_steps_per_bar=16, n_pitches=84,
                 hid_features=1024, hid_channels=512, out_channels=1):
        super().__init__()
        self.Input = nn.Sequential()
        self.Input.add_module('dense0', nn.Linear(4* z_dim, hid_features))
        self.Input.add_module('bn0', nn.BatchNorm1d(hid_features))
        self.Input.add_module('relu0', nn.ReLU())
        self.Input.add_module('reshape0',
                              Reshape(dims=[hid_channels, hid_features//hid_channels, 1]))
        
        self.Base = nn.Sequential()
        self.Base.add_module('upconv1', nn.ConvTranspose2d(hid_channels, hid_channels,
                                        kernel_size=(2, 1), stride=(2, 1), padding=0))
        self.Base.add_module('bn1', nn.BatchNorm2d(hid_channels))
        self.Base.add_module('relu1', nn.ReLU())
        
        self.Base.add_module('upconv2', nn.ConvTranspose2d(hid_channels, hid_channels//2,
                                        kernel_size=(2, 1), stride=(2, 1), padding=0))
        self.Base.add_module('bn2', nn.BatchNorm2d(hid_channels//2))
        self.Base.add_module('relu2', nn.ReLU())
        
        self.Base.add_module('upconv3', nn.ConvTranspose2d(hid_channels//2, hid_channels//2,
                                        kernel_size=(2, 1), stride=(2, 1), padding=0))
        self.Base.add_module('bn3', nn.BatchNorm2d(hid_channels//2))
        self.Base.add_module('relu3', nn.ReLU())
        
        self.Base.add_module('upconv4', nn.ConvTranspose2d(hid_channels//2, hid_channels//2,
                                        kernel_size=(1, 7), stride=(1, 7), padding=0))
        self.Base.add_module('bn4', nn.BatchNorm2d(hid_channels//2))
        self.Base.add_module('relu4', nn.ReLU())
        
        self.Output = nn.Sequential()
        self.Output.add_module('upconv5',nn.ConvTranspose2d(hid_channels//2, out_channels,
                                         kernel_size=(1, 12), stride=(1, 12), padding=0))
        self.Output.add_module('reshape5', Reshape(dims=[1, 1, n_steps_per_bar, n_pitches]))
        
    def forward(self, x):
        x = self.Input(x)
        x = self.Base(x)
        x = self.Output(x)
        return x

######################################
######       MuseCritic        #######
######################################
class MuseCritic(nn.Module):
    def __init__(self, input_shape=(4, 2, 16, 84),
                       hid_channels=128,
                       hid_features=1024,
                       out_features=1):
        super().__init__()
        n_tracks, n_bars, n_steps_per_bar, n_pitches = input_shape
        self.Input=nn.Identity()
        
        self.Base = nn.Sequential()
        self.Base.add_module('conv0', nn.Conv3d(n_tracks, hid_channels,
                                      kernel_size=(2, 1, 1), stride=(1,1,1), padding=0))
        self.Base.add_module('lrelu0', nn.LeakyReLU(0.3)) 
        
        self.Base.add_module('conv1', nn.Conv3d(hid_channels, hid_channels,
                                      kernel_size=(n_bars-1, 1, 1), stride=(1,1,1), padding=0))
        self.Base.add_module('lrelu1', nn.LeakyReLU(0.3))
        
        self.Base.add_module('conv2', nn.Conv3d(hid_channels, hid_channels,
                                      kernel_size=(1, 1, 12), stride=(1,1,12), padding=0))
        self.Base.add_module('lrelu2', nn.LeakyReLU(0.3)) 
        
        self.Base.add_module('conv3', nn.Conv3d(hid_channels, hid_channels,
                                      kernel_size=(1, 1, 7), stride=(1,1,7), padding=0))
        self.Base.add_module('lrelu3', nn.LeakyReLU(0.3)) 
        
        self.Base.add_module('conv4', nn.Conv3d(hid_channels, hid_channels,
                                      kernel_size=(1, 2, 1), stride=(1,2,1), padding=0))
        self.Base.add_module('lrelu4', nn.LeakyReLU(0.3)) 
        
        self.Base.add_module('conv5', nn.Conv3d(hid_channels, hid_channels,
                                      kernel_size=(1, 2, 1), stride=(1,2,1), padding=0))
        self.Base.add_module('lrelu5', nn.LeakyReLU(0.3)) 
        
        self.Base.add_module('conv6', nn.Conv3d(hid_channels, 2*hid_channels,
                                      kernel_size=(1, 4, 1), stride=(1,2,1), padding=(0, 1, 0)))
        self.Base.add_module('lrelu6', nn.LeakyReLU(0.3)) 
        
        self.Base.add_module('conv7', nn.Conv3d(2*hid_channels, 4*hid_channels,
                                      kernel_size=(1, 3, 1), stride=(1,2,1), padding=(0, 1, 0)))
        self.Base.add_module('lrelu7', nn.LeakyReLU(0.3))
        
        self.Output=nn.Sequential()
        self.Output.add_module('flatten', nn.Flatten())
        self.Output.add_module('linear', nn.Linear(4*hid_channels, hid_features))
        self.Output.add_module('lrelu', nn.LeakyReLU(0.3))
        self.Output.add_module('fc', nn.Linear(hid_features, out_features))
        
    def forward(self, x):
        x = self.Input(x)
        x = self.Base(x)
        x = self.Output(x)
        return x

######################################
######       MuseGenerator     #######
######################################
class MuseGenerator(nn.Module):
    def __init__(self, n_tracks=4, n_bars=2, n_steps_per_bar=16, n_pitches=84,
                       z_dim=32, hid_channels=1024, hid_features=1024, out_channels=1):
        super().__init__()
        self.n_bars=n_bars
        self.n_tracks=n_tracks
        
        self.ChordsNetwork=TempNetwork(z_dim=z_dim,
                                     n_bars=n_bars,
                                     hid_channels=hid_channels)
        
        self.MelodyNetworks=nn.ModuleDict({})
        for n in range(n_tracks):
            self.MelodyNetworks.add_module('melodygen'+str(n), TempNetwork(z_dim=z_dim,
                                                              n_bars=n_bars,
                                                              hid_channels=hid_channels))
        self.BarGenerators=nn.ModuleDict({})   
        for n in range(n_tracks):
            self.BarGenerators.add_module('bargen'+str(n), BarGenerator(z_dim=z_dim,
                                                    n_steps_per_bar=n_steps_per_bar,
                                                    n_pitches=n_pitches,
                                                    hid_features=hid_features,
                                                    hid_channels=hid_channels//2,
                                                    out_channels=out_channels))
        
    def forward(self, chords, style, melody, groove):
        # Chords ==> (N * dimZ)
        # Style  ==> (N * dimZ)
        # Melody ==> (N * nTracks * dimZ)
        # Groove ==> (N * nTracks * dimZ)
        chordOuts = self.ChordsNetwork(chords)
        barOuts=[]
        for bar in range(self.n_bars):
            trackOuts=[]
            chordOut = chordOuts[:, :, bar]
            styleOut = style
            for track in range(self.n_tracks):
                melodyInp = melody[:, track, :]
                melodyOut = self.MelodyNetworks['melodygen'+str(track)](melodyInp)[:, :, bar]
                grooveOut = groove[:, track, :]
                z = torch.cat([chordOut, styleOut, melodyOut, grooveOut], dim=1)
                trackOuts.append(self.BarGenerators['bargen'+str(track)](z))
            trackOut = torch.cat(trackOuts, dim=1)
            barOuts.append(trackOut)
        out = torch.cat(barOuts, dim=2)
        # Out ==> (N * nTracks * nBars * nStepsPerBar * nPitches)
        return out