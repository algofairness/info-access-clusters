import os

def main():
    path = 'input/test_input/getalphas/'

    for file in os.listdir(path):
        oldname=file
        newname=file[0:-4]+'_'+file[-4:]
        os.rename(path+oldname, path+newname)
    return

if __name__ == "__main__":
    main()
