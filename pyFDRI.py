"""
    This is a module for python3 program which reconstructs an image sampled compressively 
    with binarized DCT functions using the FDRI method.

    Code from:
    Krzysztof M. Czajkowski K.M., Anna Pastuszczak A., and Rafał Kotyński
    "Real-time single-pixel video imaging with Fourier domain regularization,"
    Optics Express, vol. 26(16), pp. 20009-20022, (2018).
    http://dx.doi.org/10.1364/OE.26.020009

    This code is adapted from the matlab FDRI package provided at
    https://www.igf.fuw.edu.pl/fdri

    Adapted by: G.G Stoltz
    Date      : 09/2018
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

"""
    Usage: See Example in Main.

    To run example in terminal:
        python3 pyFDRI.py

    Log:
    09/2018: Initial. Note: From a python perspective there is still changes that can be made
             to make the adaptation more pythonic.
"""

import numpy as np
import matplotlib.pylab as plt
from scipy.fftpack import dct
from scipy.fftpack import idct
from scipy.sparse import spdiags
import imageio
import skimage.transform
import skimage
import time

def dct2(x):
    """
    DCT is applied to rows of X. DCT is applied again to the columns of the resulting matrix
    **Can change axis of dct
    """
    return dct(dct(x, norm='ortho', axis=1), norm='ortho', axis=0)

def idct2(x):
    """
    DCT is applied to rows of X. DCT is applied again to the columns of the resulting matrix
    **Can change axis of dct
    """
    return idct(idct(x, norm='ortho', axis=1), norm='ortho', axis=0)

def AvgDCTSpectrum(dim, img_path, warning=True):
    """
    Calculate the magnitude of the DCT spectrum averaged over an image database
    img_path is the path to images and images are called 'n.gif' with n=1..49
    Images are first resized to size dim 

        dim      - tuple of (Ny, Nx)
        img_path - folder path for average DCT calculation.

    """
    try:
        N = 49 # image count
        AvgDCT = np.zeros(dim)
        for ii in range(N):
            im = imageio.imread("{}{}{}".format(img_path,ii,".gif")).astype(np.float)
            im = im / im.max() # Scale to 0-1.
            x = skimage.transform.resize(im,dim,mode='constant')
            AvgDCT = AvgDCT + np.abs(dct2(x))
        
        AvgDCT = AvgDCT / N
        
    except:
        if (warning):
            print("WARNING: cannot read the image database." + 
              "Please make sure to put a set of {} test images (1.gif, 2.gif...)".format(N) +
              "in the {} folder.".format(img_path))
            print("... continuing with a simple model for the average DCT spectrum...")
        AvgDCT = np.zeros(dim)
        [x,y] = np.meshgrid(np.linspace(1e-3,1,dim[1]),np.linspace(1e-3,1,dim[0]))
        AvgDCT = 1/(x + y)
        AvgDCT = AvgDCT / AvgDCT[0,0]
        
    return AvgDCT

def SelectionMatrix(k, AvgDCT, deterministic=False):
    """
    Create a (logical) selection matrix SM for the DCT basis
    with k true elements at the selected DCT basis
    Selection is random with with the Bernoulli probabilities proportional to AvgDCT 
    """
    if (deterministic):
        R = AvgDCT 
    else:
        R = np.random.random(AvgDCT.shape) * AvgDCT
    # Add the zeroth frequency into selection
    R[0,0] = AvgDCT[0,0] 
    Idx = np.argsort(R.flat)
    SM = np.zeros(AvgDCT.shape,np.bool).flatten()
    SM[Idx[-k:]] = True
    SM = SM.reshape(AvgDCT.shape)
    return SM

def MeasurementMatrix(SM,binarize=False):
    """
    Return the measurement matrix M with rows containing DCT basis selected 
    by the elements of selection matrix SM
    """
    dim = SM.shape         
    m_rows = np.sum(SM)    # Select number of sample points to be made
    m_cols = SM.size       # Size of measurement, also total elements in image.
    M = np.zeros([m_rows,m_cols]) # Create Measreument Matrix
    P = np.arange(SM.size)
    P = P[SM.flat]

    # Put the respective DCT basis in every row of M
    for r in range(m_rows):
        eye = np.zeros(dim)
        eye.flat[P[r]] = 1.0
        M[r,:] = idct2(eye).flatten()
        if binarize:
            M[r,:] = (M[r,:] >= M[r,:].mean()).astype(np.float)
    return M

def fdri(M,Nx,Ny,mi=0.5,ep=1e-5,method=0):
    """
     Calculates the generalised inverse of the measurement matrix using the FDRI method
     Czajkowski et al, Opt. Express 26, 20009, 2018, http://dx.doi.org/10.1364/OE.26.020009
    
     Input parameters:
     M - measurement matrix. Every row includes one Nx x Ny sampling function
     mi - FDRI parameter (defaults to mi=0.5)
     ep - FDRI parameter (defaults to ep=1e-5)
     method =0 (default) calculate P with Eq. (7)
     method =1 calculate P with Eq. (8) using the pinv function
     method =2 calculate P with Eq. (8) using svd
    
     Output parameters:
     P - the generalized inverse matrix (calculated with Eq. (7) or (8))
    
     Krzysztof M. Czajkowski K.M., Anna Pastuszczak A., and Rafał Kotyński
     "Real-time single-pixel video imaging with Fourier domain regularization,"
     Optics Express, vol. 26(16), pp. 20009-20022, (2018).
     http://dx.doi.org/10.1364/OE.26.020009
    """
    k = M.size / Nx / Ny
    M = np.reshape(M,(int(k), Nx*Ny))

    # Calculate the diagonal elements of hat(Gamma) according to Eq. (11)
    ry = np.hstack((np.arange(0,Ny/2), np.arange(-Ny/2,0)))
    rx = np.hstack((np.arange(0,Nx/2), np.arange(-Nx/2,0)))  
    wx, wy = np.meshgrid(2*np.pi/Nx*rx, 2*np.pi/Ny*ry)
    D = 1.0 / np.sqrt((1 - mi)**2 * (np.sin(wx)**2 + np.sin(wy)**2) + ep + mi**2 * (wx**2 + wy**2) / (2*np.pi**2))

    # Helper functions - apply 2D DFT to images stored in rows or columns of a matrix X
    row_fft2 = lambda X: np.fft.fft2(X.reshape((-1,Ny,Nx))).reshape((-1,Ny*Nx))       # size of X is [k, n*n], fft2 is applied to rows
    row_ifft2 = lambda X: np.fft.ifft2(X.reshape((-1,Ny,Nx))).reshape((-1,Ny*Nx))     # size of X is [k, n*n], ifft2 is applied to rows
    col_fft2 = lambda X: np.fft.fft2(X.T.reshape((-1,Ny,Nx))).reshape((-1,Ny*Nx)).T   # size of X is [n*n,k], fft2 is applied to columns
    col_ifft2 = lambda X: np.fft.ifft2(X.T.reshape((-1,Ny,Nx))).reshape((-1,Ny*Nx)).T # size of X is [n*n,k], ifft2 is applied to columns
    
    # Helper functions - apply 2D linear filtering to a matrix X
    FILT_R = lambda X: row_fft2(row_ifft2(X) @ spdiags(D.flat,0,Nx*Ny,Nx*Ny)) # X * F' * D  * F
    FILT_L = lambda X: col_fft2(spdiags(D.flat,0,Nx*Ny,Nx*Ny) @ col_ifft2(X)) # F * D  * F' * X

    # Now calculate the generalized inverse matrix P
    a = np.real(FILT_R(M))
    if (method == 1):
    # Calculate the inversion matrix with Eq. (8)
        P = FILT_L(np.linalg.pinv(a))
    elif (method == 2):
    # Use SVD to calculate the pseudoinverse, and then use Eq. (8):
        U, S, Vh = np.linalg.svd (a,0) #a = U*S*Vh
        P = FILT_L( np.asarray(np.asmatrix(Vh).getH()) @ np.diag(1/S) @ np.asarray(np.asmatrix(U).getH()))
    else:
    # Calculate the inversion matrix with Eq. (7) - default
        a = np.asmatrix(a)
        P = np.real(FILT_L(np.asarray(a.getH() @ (np.linalg.inv(a @ a.getH())))))
    return P

if __name__ == "__main__":
    print("Fourier Domain Regularized Inversion (FDRI) example.\n")
    # All images will be resized to NxN pixels
    N = 256          

    # Number of basis functions (the compression ratio is equal to k/N^2)
    k = 1966         

    # FDRI parameter mi (Eq. 11)
    mi = 0.5         

    # Use binarized or continuous DCT functions for the measurement matrix 
    binarize = True  

    # Path to images
    img_path = "./images/"

    # Path to the test image
    images = ["bird512.jpg"]

    scene_path = img_path + images[0]

    print("Image input: {}".format(scene_path))
    print("Resolution: [{} x {}]".format(N,N))
    print("Compression ratio: {:.3f}".format( k / N**2 * 100 ))

    dim = (N, N)

    print("\nI. Preparation Stage")
    print("1. calculate the average DCT spectrum using an image database...")
    AvgDCT = AvgDCTSpectrum(dim, img_path,False)

    print("2. Select {} DCT bases randomly with probabilities ".format(k) + 
          "proportional to average occurances in the image database...")
    SM = SelectionMatrix(k,AvgDCT)

    print("3. Prepare the {} x {} measurement".format(k, np.prod(dim)) + 
          "matrix consisting of the selected binarized DCT functions...")
    M = MeasurementMatrix(SM, binarize)

    print("4. Calculate the generalized inverse matrix with FDRI" + 
          "(takes some time)...")
    P = fdri(M,dim[1],dim[0],mi)

    print("\nII. COMPRESSIVE MEASUREMENT")
    print("1. Prepare the scene...")
    im = imageio.imread(scene_path).astype(np.float)
    im = im / im.max()
    xorig = skimage.transform.resize(im,dim,mode='constant')
    x = xorig.flatten()

    print("2. Take the compressive measurement...")
    y = M @ x # Eq. (1)

    print("3. Reconstruct the image with FDRI...")
    t1 = time.time()
    x0 = P @ y
    t2 = time.time()
    x0 = np.reshape(x0,(N,N))
    print("   time={:.3f} ms".format((t2-t1)*1000))

    fig, ax = plt.subplots(2,2,figsize=(12,12))

    fig.suptitle("FDRI Reconstruction Method 0 (default) calculate P with Eq. (7)")

    ax[0,0].set_title("Reference image")
    ax[0,0].imshow(xorig, cmap='gray',interpolation='none',vmin=0, vmax=1.0)

    ax[0,1].set_title("Selected DCT basis")
    ax[0,1].imshow(SM.astype(np.float), cmap='gray',interpolation='none',vmin=0, vmax=1.0)

    ax[1,0].set_title("Measured data (compr. ratio= {:.2f})".format(100*k/N**2))
    ax[1,0].plot(y, '.r')
    ax[1,0].set_xlim([0,y.size])

    ax[1,1].set_title("Reconstructed image (FDRI,mi={})".format(mi))
    ax[1,1].imshow(np.real(x0), cmap='gray', interpolation='none',vmin=0, vmax=1.0)

    plt.show()
