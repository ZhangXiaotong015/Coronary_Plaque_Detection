from torchvision.ops import misc

def conv2group(backbone):
    'Modify the parameter groups in each convlutional layer'
    group_num = 7
    # for name, parameter in backbone.body.named_modules(): print(name)
    for name, parameter in backbone.body.named_modules():
        if 'layer' in name and len(name.split('.'))==2:
            # for name2, parameter in backbone.body[name.split('.')[0]][int(name.split('.')[1])].named_modules():
            in_channels = backbone.body[name.split('.')[0]][int(name.split('.')[1])].conv1.in_channels
            out_channels = backbone.body[name.split('.')[0]][int(name.split('.')[1])].conv1.out_channels
            groups = backbone.body[name.split('.')[0]][int(name.split('.')[1])].conv1.groups
            backbone.body[name.split('.')[0]][int(name.split('.')[1])].conv1.in_channels = in_channels*group_num
            backbone.body[name.split('.')[0]][int(name.split('.')[1])].conv1.out_channels = out_channels*group_num
            backbone.body[name.split('.')[0]][int(name.split('.')[1])].conv1.groups = group_num
            backbone.body[name.split('.')[0]][int(name.split('.')[1])].bn1 = misc.FrozenBatchNorm2d(out_channels*7)
            backbone.body[name.split('.')[0]][int(name.split('.')[1])].conv1.weight.data = backbone.body[name.split('.')[0]][int(name.split('.')[1])].conv1.weight.data.repeat(group_num,1,1,1)
            # backbone.body[name.split('.')[0]][int(name.split('.')[1])].bn1.weight.data = backbone.body[name.split('.')[0]][int(name.split('.')[1])].bn1.weight.data.repeat(group_num)

            in_channels = backbone.body[name.split('.')[0]][int(name.split('.')[1])].conv2.in_channels
            out_channels = backbone.body[name.split('.')[0]][int(name.split('.')[1])].conv2.out_channels
            groups = backbone.body[name.split('.')[0]][int(name.split('.')[1])].conv2.groups
            backbone.body[name.split('.')[0]][int(name.split('.')[1])].conv2.in_channels = in_channels*group_num
            backbone.body[name.split('.')[0]][int(name.split('.')[1])].conv2.out_channels = out_channels*group_num
            backbone.body[name.split('.')[0]][int(name.split('.')[1])].conv2.groups = group_num
            backbone.body[name.split('.')[0]][int(name.split('.')[1])].bn2 = misc.FrozenBatchNorm2d(out_channels*7)
            backbone.body[name.split('.')[0]][int(name.split('.')[1])].conv2.weight.data = backbone.body[name.split('.')[0]][int(name.split('.')[1])].conv2.weight.data.repeat(group_num,1,1,1)
            # backbone.body[name.split('.')[0]][int(name.split('.')[1])].bn2.weight.data = backbone.body[name.split('.')[0]][int(name.split('.')[1])].bn2.weight.data.repeat(group_num)

            in_channels = backbone.body[name.split('.')[0]][int(name.split('.')[1])].conv3.in_channels
            out_channels = backbone.body[name.split('.')[0]][int(name.split('.')[1])].conv3.out_channels
            groups = backbone.body[name.split('.')[0]][int(name.split('.')[1])].conv3.groups
            backbone.body[name.split('.')[0]][int(name.split('.')[1])].conv3.in_channels = in_channels*group_num
            backbone.body[name.split('.')[0]][int(name.split('.')[1])].conv3.out_channels = out_channels*group_num
            backbone.body[name.split('.')[0]][int(name.split('.')[1])].conv3.groups = group_num
            backbone.body[name.split('.')[0]][int(name.split('.')[1])].bn3 = misc.FrozenBatchNorm2d(out_channels*7)
            backbone.body[name.split('.')[0]][int(name.split('.')[1])].conv3.weight.data = backbone.body[name.split('.')[0]][int(name.split('.')[1])].conv3.weight.data.repeat(group_num,1,1,1)
            # backbone.body[name.split('.')[0]][int(name.split('.')[1])].bn3.weight.data = backbone.body[name.split('.')[0]][int(name.split('.')[1])].bn3.weight.data.repeat(group_num)

            if int(name.split('.')[1])==0:
                in_channels = backbone.body[name.split('.')[0]][int(name.split('.')[1])].downsample[0].in_channels*group_num
                out_channels = backbone.body[name.split('.')[0]][int(name.split('.')[1])].downsample[0].out_channels*group_num
                groups = backbone.body[name.split('.')[0]][int(name.split('.')[1])].downsample[0].groups
                backbone.body[name.split('.')[0]][int(name.split('.')[1])].downsample[0].in_channels = in_channels
                backbone.body[name.split('.')[0]][int(name.split('.')[1])].downsample[0].out_channels = out_channels
                backbone.body[name.split('.')[0]][int(name.split('.')[1])].downsample[0].groups = group_num
                backbone.body[name.split('.')[0]][int(name.split('.')[1])].downsample[1] = misc.FrozenBatchNorm2d(out_channels)
                backbone.body[name.split('.')[0]][int(name.split('.')[1])].downsample[0].weight.data = backbone.body[name.split('.')[0]][int(name.split('.')[1])].downsample[0].weight.data.repeat(group_num,1,1,1)
                # backbone.body[name.split('.')[0]][int(name.split('.')[1])].downsample[1].weight.data = backbone.body[name.split('.')[0]][int(name.split('.')[1])].downsample[1].weight.data.repeat(group_num)

    return backbone