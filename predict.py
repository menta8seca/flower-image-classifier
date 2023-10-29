from utility import get_predict_args,translate,display_result
from model import recover_model,predict

def main():
    input_args=get_predict_args()
    
    model,idx_to_cls=recover_model(input_args.checkpoint)
    
    ps,classes=predict(input_args.image,model,input_args.top_k,input_args.gpu)
    t_classes=translate(classes,idx_to_cls,input_args.category_names)
    print(ps)
    print(t_classes)
    print(classes)
    display_result(ps,t_classes)
    
if __name__ == "__main__":
    main()
    