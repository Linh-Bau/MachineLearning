import numpy as np

# A=[img_of_row,(img_w*img_h)*img_of_column]
# =>
# img_h*img_of_row, img_w*img_of_column
def convert_matrix(matrix, img_w, img_h):
    if matrix.shape[1]%img_w!=0:
        print('img_w-img_h not correct! col-w-h {}:{}:{}'.format(matrix.shape[1],img_w,img_h))
        return
    
    img_of_row=int(matrix.shape[1]/(img_w*img_h))
    img_of_col=int(matrix.shape[0])
    M=None # M= [img_h*img_of_row, img_w*img_of_column]
    for i in range(img_of_col):
        row_data=matrix[i,:] # [(img_h*img_w)*img_of_row] must convert to [img_h, img_w*img_of_row]
        img_row=np.zeros((img_h,img_w*img_of_row))
        for index in range(len(row_data)):
            img_index=int(index/(img_w*img_h))
            r=index-img_index*(img_w*img_h)
            row=int(r/img_w)
            col=r-row*img_w
            img_row[row,img_index*img_w+col]=row_data[index]
        if M is None:
            M=img_row
        else:
            M=np.vstack((M,img_row))
    return M

def convert_matrix_of_color_img(matrix,img_w,img_h):
    convert_matrix(np.reshape(matrix[:,0],(1,-1)),img_w,img_h)
    convert_matrix(np.reshape(matrix[:,1],(1,-1)),img_w,img_h)
    convert_matrix(np.reshape(matrix[:,2],(1,-1)),img_w,img_h)

if __name__=='__main__':
    #picture 2x2
    #3 picture of row, 2 picture of column
    # map
    # 1 2 3
    # 4 5 6
    # matrix map
    # 1 1 1 1 2 2 2 2 3 3 3 3
    # 4 4 4 4 5 5 5 5 6 6 6 6
    matrix=np.array([[11,12,13,14,15,16,17,18,19,],[21,22,23,24,25,26,27,28,29,]])
    converted=convert_matrix(matrix,3,3)
    print(matrix)
    print('------')
    print(converted)
    