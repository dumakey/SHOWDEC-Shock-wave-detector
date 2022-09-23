import re

def read_case_setup(launch_filepath):
    file = open(launch_filepath, 'r')
    data = file.read()

    class setup:
        pass

    casedata = setup()
    casedata.case_dir = None
    casedata.analysis = dict.fromkeys(['type', 'import', 'augD_ID'], None)
    casedata.training_parameters = \
        dict.fromkeys(['train_size', 'learning_rate', 'l2_reg', 'l1_reg', 'dropout', 'epochs', 'batch_size'], None)
    casedata.training_parameters.update({'addaugdata': [None, None]})
    casedata.img_processing = {'slice_size': [None, None],
                               'rotation': [None, None, None, None],
                               'translation': [None, None, None],
                               'zoom': [None, None],
                               'filter': [None, None, None, None, None],
                               'flip': [None, None]
                               }
    casedata.data_augmentation = [None, None]

    ## Data directory
    match = re.search('DATADIR\s*=\s*(.*).*', data)
    if match:
        casedata.case_dir = match.group(1)

    ## Analysis
    # Type of analysis
    match = re.search('TYPEANALYSIS\s*=\s*(\w+).*', data)
    if match:
        casedata.analysis['type'] = match.group(1)

    # Import
    match = re.search('IMPORTMODEL\s*=\s*(\d).*', data)
    if match:
        casedata.analysis['import'] = int(match.group(1))

    # Augmented dataset selector
    match = re.search('AUGDATASETID\s*=\s*(\d).*', data)
    if match:
        casedata.analysis['augD_ID'] = int(match.group(1))

    ## Dataset augmentation
    match = re.search('AUGDATA\s*=\s*(\d).*', data)
    if match:
        casedata.data_augmentation[0] = int(match.group(1))
        match_factor = re.search('AUGDATASIZE\s*=\s*(\d+\.?\d*).*', data)
        if match_factor:
            casedata.data_augmentation[1] = float(match_factor.group(1))

    ## Training parameters
    # Training dataset size
    match = re.search('TRAINSIZE\s*=\s*(\d+\.?\d*|NONE).*', data)
    if match:
        if match.group(1) == 'NONE':
            casedata.training_parameters['train_size'] = 0.75
        else:
            casedata.training_parameters['train_size'] = float(match.group(1))

    # Learning rate
    match = re.search('LEARNINGRATE\s*=\s*(\d+\.?\d*|NONE).*', data)
    if match:
        if match.group(1) == 'NONE':
            casedata.training_parameters['learning_rate'] = 0.001
        else:
            casedata.training_parameters['learning_rate'] = float(match.group(1))

    # L2 regularizer
    match = re.search('L2REG\s*=\s*(\d+\.?\d*|NONE).*', data)
    if match:
        if match.group(1) == 'NONE':
            casedata.training_parameters['l2_reg'] = 0.0
        else:
            casedata.training_parameters['l2_reg'] = float(match.group(1))

    # L1 regularizer
    match = re.search('L1REG\s*=\s*(\d+\.?\d*|NONE).*', data)
    if match:
        if match.group(1) == 'NONE':
            casedata.training_parameters['l1_reg'] = 0.0
        else:
            casedata.training_parameters['l1_reg'] = float(match.group(1))

    # Dropout
    match = re.search('DROPOUT\s*=\s*(\d+\.?\d*|NONE).*', data)
    if match:
        if match.group(1) == 'NONE':
            casedata.training_parameters['dropout'] = 0.0
        else:
            casedata.training_parameters['dropout'] = float(match.group(1))

    # Number of epochs
    match = re.search('EPOCHS\s*=\s*(\d+\.?\d*|NONE).*', data)
    if match:
        if match.group(1) == 'NONE':
            casedata.training_parameters['epochs'] = 1
        else:
            casedata.training_parameters['epochs'] = int(match.group(1))

    # Batch size
    match = re.search('BATCHSIZE\s*=\s*(\d+\.?\d*|NONE).*', data)
    if match:
        if match.group(1) == 'NONE':
            casedata.training_parameters['batch_size'] = None
        else:
            casedata.training_parameters['batch_size'] = int(match.group(1))

    # Add (existing) augmented dataset
    match = re.search('ADDAUGMENTED\s*=\s*(\d).*', data)
    if match:
        casedata.training_parameters['addaugdata'][0] = int(match.group(1))
        match_factor = re.search('ADDAUGDATAID\s*=\s*(\d+\.?\d*).*', data)
        if match_factor:
            casedata.training_parameters['addaugdata'][1] = int(match_factor.group(1))
    ## Image processing parameters
    # Image resize
    match_dist = re.search('IMAGERESIZE\s*=\s*\((\d+|NONE)\,+(\d+|NONE)\).*', data)
    if match_dist:
        casedata.img_processing['slice_size'][0] = int(match_dist.group(1))
        casedata.img_processing['slice_size'][1] = int(match_dist.group(2))
        casedata.img_processing['slice_size'] = tuple(casedata.img_processing['slice_size'])

    # Rotation
    match = re.search('ROTATION\s*=\s*(\d).*', data)
    if match:
        casedata.img_processing['rotation'][0] = int(match.group(1))
        match = re.search('ROTATIONCENTER\s*=\s*\((\d+|NONE)\,+(\d+|NONE)\).*', data)
        if match:
            if match.group(1) != 'NONE':
                casedata.img_processing['rotation'][1] = int(match.group(1))
            elif match.group(2) != 'NONE':
                casedata.img_processing['rotation'][2] = int(match.group(2))
            match_angle = re.search('ROTATIONANGLE\s*=\s*(\d+\.?\d*).*', data)
            if match_angle:
                casedata.img_processing['rotation'][3] = float(match_angle.group(1))

    # Translation
    match = re.search('TRANSLATION\s*=\s*(\d).*', data)
    if match:
        casedata.img_processing['translation'][0] = int(match.group(1))
        match_dist = re.search('TRANSLATIONDIST\s*=\s*\((\d+|NONE)\,+(\d+|NONE)\).*', data)
        if match_dist:
            casedata.img_processing['translation'][1] = float(match_dist.group(1))
            casedata.img_processing['translation'][2] = float(match_dist.group(2))

    # Zoom
    match = re.search('ZOOM\s*=\s*(\d).*', data)
    if match:
        casedata.img_processing['zoom'][0] = int(match.group(1))
        match_factor = re.search('ZOOMFACTOR\s*=\s*(\d+\.?\d*).*', data)
        if match_factor:
            casedata.img_processing['zoom'][1] = float(match_factor.group(1))
    # Filter
    match = re.search('FILTER\s*=\s*(\d).*', data)
    if match:
        casedata.img_processing['filter'][0] = int(match.group(1))
        match_type = re.search('FILTERTYPE\s*=\s*(\w+).*', data)
        casedata.img_processing['filter'][1] = match_type.group(1)
        if match_type:
            if match_type.group(1) == 'GAUSSIAN':
                filter_param = re.search(
                    'FILTERPARAM\s*=\s*\(\s*SIZE\s*\,\s*(\d+|NONE)\s*\,\s*SIGMA\s*\,\s*(\d+|NONE)\s*\).*', data)
                casedata.img_processing['filter'][2] = int(filter_param.group(1))
                casedata.img_processing['filter'][3] = int(filter_param.group(2))
        elif match_type.group(1) == 'BILATERAL':
            filter_param = re.search(
                'FILTERPARAM\s*=\s*\(\s*(D)\s*\,\s*(\d+|NONE)\s*\,\s*SIGMACOLOR\s*\,\s*(\d+|NONE)\s*SIGMASPACE\s*\,\s*(\d+|NONE)\s*\).*',
                data)
            casedata.img_processing['filter'][2] = int(filter_param.group(1))
            casedata.img_processing['filter'][3] = int(filter_param.group(2))
            casedata.img_processing['filter'][4] = int(filter_param.group(3))

    # Flip
    match = re.search('FLIP\s*=\s*(\d).*', data)
    if match:
        casedata.img_processing['flip'][0] = int(match.group(1))
        match_type = re.search('FLIPTYPE\s*=\s*(\w+).*', data)
        if match_type:
            casedata.img_processing['flip'][1] = match_type.group(1)

    return casedata