
Three ways to view/download the images from HPC.

1. If you use VSCode, you can simply view the original and processed images in the file explorer.
2. If you have set up sftp GUI clients like WinSCP/XFTP, you can use it to download the images from HPC.
3. Alternatively, you may use scp commands in your local terminal to download the images. For example, to download the folder `your_hpc_path` from HPC to your local path `your_local_download_path`, you can use the following command:

   scp -r <your_crsid>@login.hpc.cam.ac.uk:~/your_hpc_path ~/your_local_download_path

