from .resnet import resnet101, resnet50, resnet18

def deserialize_network(dic, pretrain=False):
    typ = dic['_type']
    if typ == 'resnet101':
        num_outputs = dic['num_outputs']
        if 'conv_at_end' not in dic:
            print('WARNING forgot conv_at_end in resnet config')
            conv_at_end = True
        else:
            conv_at_end = dic['conv_at_end']
        return resnet101(pretrain, True, num_classes=num_outputs, conv_at_end=conv_at_end)
    elif typ == 'resnet50':
        num_outputs = dic['num_outputs']
        if 'conv_at_end' not in dic:
            print('WARNING forgot conv_at_end in resnet config')
            conv_at_end = True
        else:
            conv_at_end = dic['conv_at_end']
        return resnet50(pretrain, True, num_classes=num_outputs, conv_at_end=conv_at_end)
    elif typ == 'resnet18':
        num_outputs = dic['num_outputs']
        if 'conv_at_end' not in dic:
            print('WARNING forgot conv_at_end in resnet config')
            conv_at_end = True
        else:
            conv_at_end = dic['conv_at_end']
        return resnet18(pretrain, True, num_classes=num_outputs, conv_at_end=conv_at_end)

    else:
        raise Exception("Failed deserialization of {}".format(dic))


def serialize_network(net):
    num_outputs = net.head.weight.shape[0]
    conv_at_end = net.conv_at_end
    type_name = net.type_name
    return {'_type': type_name,
            'num_outputs': num_outputs,
            'conv_at_end': conv_at_end
            }

