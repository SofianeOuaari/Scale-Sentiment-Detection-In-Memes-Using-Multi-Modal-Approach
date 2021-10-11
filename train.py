from preprocess import return_rgb_images_text,preprocess_text,label_encoder
from models import Unimodal










if __name__=="__main__":

    img_arr,img_name,texts,y=return_rgb_images_text(224,224)
    tokenized_texts,tokenizer=preprocess_text(texts,500,25,"text_tokenizer.pickle")

    unimo=Unimodal()
    vocab_size=len(tokenizer.word_index)+1

    print(unimo.word_embedding_lstm(tokenized_texts,vocab_size,300,5,5,3,is_gpu_available=True).summary())
    
    encoded_label,l_e=label_encoder(y)

    w_e_lstm=unimo.word_embedding_lstm(tokenized_texts,vocab_size,300,5,5,5,is_gpu_available=True)

    w_e_lstm.fit(tokenized_texts,encoded_label,batch_size=32,epochs=15,validation_split=0.25)

    '''cnn_model=unimo.image_cnn_model(img_arr,2,2,5,(3,3),(2,2),number_labels=5)

    cnn_model.fit(img_arr,encoded_label,batch_size=8,epochs=15,validation_split=0.25)'''

