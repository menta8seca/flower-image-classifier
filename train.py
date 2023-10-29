from utility import get_train_args,prepaire_data
from model import build_model,train_model,test_model,save_model

def main():
    input_args=get_train_args()
    
    tr_loader,v_loader,ts_loader,cls_to_idx=prepaire_data(input_args.dir)
    model,criterion,optimizer=build_model(input_args.arch,input_args.learning_rate,input_args.hidden_units)
    
    train_model(tr_loader,v_loader,model,criterion,optimizer,input_args.epochs,input_args.gpu)
    test_model(ts_loader,model,criterion,input_args.gpu)
    model.class_to_idx = cls_to_idx
    
    save_model(model.classifier.state_dict(),optimizer.state_dict(),cls_to_idx,input_args.save_dir,input_args.epochs,input_args.arch,input_args.learning_rate,input_args.hidden_units)
    
if __name__ == "__main__":
    main()
    