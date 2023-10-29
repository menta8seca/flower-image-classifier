from torchvision import models
import torch
from torch import nn
from torch import optim
from utility import process_image,imshow
import PIL

def build_model(arch,lr,hid):
    if arch=='densenet121':
        model=eval(f"models.{arch}(pretrained=True)")
        num_features = model.classifier.in_features
    elif arch=='vgg16':
        model=eval(f"models.{arch}(pretrained=True)")
        num_features = model.classifier[0].in_features
    else:
        print('invalid arch: only vgg16 and densenet121 are supported')
        return 0
    for par in model.parameters():
        par.requires_grad=False
    classifier = nn.Sequential(
        nn.Linear(num_features, hid),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(hid, hid),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(hid, 102),
        nn.LogSoftmax(dim=1)
    )
    model.classifier=classifier

    criterion=nn.NLLLoss()
    optimizer=optim.Adam(model.classifier.parameters(),lr=lr)
    
    return model,criterion,optimizer

def train_model(trainloader,validloader,model,criterion,optimizer,epochs,gpu):
    if torch.cuda.is_available():
        print('GPU is Avalible')
    else:
        print('GPU is NOT Avalible')
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device=torch.device("cpu")
    model.to(device)

    running_loss=0
    steps=0
    interval=5
    for i in range(epochs):
        for images,labels in trainloader:
            steps+=1
            images,labels=images.to(device),labels.to(device)
            optimizer.zero_grad()
            logps=model(images)
            loss=criterion(logps,labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            if steps%interval == 0:
                test_loss=0
                accuracy=0
                model.eval()
                with torch.no_grad():
                    for images,labels in validloader:
                        images,labels=images.to(device),labels.to(device)
                        log_ps=model(images)
                        test_loss+=criterion(log_ps,labels).item()
                        ps=torch.exp(log_ps)
                        top_p,top_class=ps.topk(1,dim=1)
                        equals= top_class==labels.view(*top_class.shape)
                        accuracy+=torch.mean(equals.type(torch.FloatTensor)).item()
                print(f"Epoch {i+1}/{epochs}.. "
                    f"Train loss: {running_loss/interval:.3f}.. "
                    f"Validation loss: {test_loss/len(validloader):.3f}.. "
                    f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()

def test_model(testloader,model,criterion,gpu):
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device=torch.device("cpu")
        
    test_loss=0
    accuracy=0
    model.eval()
    with torch.no_grad():
        for images,labels in testloader:
            images,labels=images.to(device),labels.to(device)
            log_ps=model(images)
            test_loss+=criterion(log_ps,labels).item()
            ps=torch.exp(log_ps)
            top_p,top_class=ps.topk(1,dim=1)
            equals= top_class==labels.view(*top_class.shape)
            accuracy+=torch.mean(equals.type(torch.FloatTensor)).item()
            
    print(f"Testing loss: {test_loss/len(testloader):.3f}.. "
        f"Validation accuracy: {accuracy/len(testloader):.3f}")
    
def save_model(model,optimizer,cls_to_idx,save_dir,epochs,arch,lr,hid):
    index_tooo_class = {v: k for k, v in cls_to_idx.items()}
    checkpoint = {
        'classifier_state_dict': model,
        'optimizer_state_dict': optimizer,
        'epochs': epochs,
        'class_to_idx': cls_to_idx,
        'idx_to_class': index_tooo_class,
        'arch':arch,
        'lr':lr,
        'hid':hid
    }
    torch.save(checkpoint, save_dir)
    
def recover_model(chk):
    checkpoint = torch.load(chk,map_location=torch.device('cpu'))
    model,cri,opt=build_model(checkpoint['arch'],checkpoint['lr'],checkpoint['hid'])

    model.classifier.load_state_dict(checkpoint['classifier_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model,checkpoint['idx_to_class']

def predict(img_path,model,topk,gpu):
    imgg= PIL.Image.open(img_path)
    prepaired_image=process_image(imgg)
    imshow(prepaired_image)
    
    model.eval()
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device=torch.device("cpu")
    model.to(device)
    
    input_tensor = prepaired_image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        logp = model.forward(input_tensor)
    p=torch.exp(logp)
    
    top_p,top_class=p.topk(topk,dim=1)
    top_p = top_p.squeeze().tolist()
    top_class = top_class.squeeze().tolist()
    
    if gpu and torch.cuda.is_available():
        top_p = [float(p) for p in top_p]
        top_class = [int(c) for c in top_class]
    
    return top_p,top_class