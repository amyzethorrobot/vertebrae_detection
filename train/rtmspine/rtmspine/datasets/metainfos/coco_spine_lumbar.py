dataset_info = dict(
    dataset_name='coco_spine',
    paper_info=dict(
        author='Lin, Tsung-Yi and Maire, Michael and '
        'Belongie, Serge and Hays, James and '
        'Perona, Pietro and Ramanan, Deva and '
        r'Doll{\'a}r, Piotr and Zitnick, C Lawrence',
        title='Microsoft coco: Common objects in context',
        container='European conference on computer vision',
        year='2014',
        homepage='http://cocodataset.org/',
    ),
    keypoint_info={
        0:
        dict(
            name='L1TL',
            id=0,
            color=[255, 0, 0],
            type='upper',
            swap=''),
        1:
        dict(
            name='L1TR',
            id=1,
            color=[255, 0, 0],
            type='upper',
            swap=''),
        2:
        dict(
            name='L1BL',
            id=2,
            color=[255, 0, 0],
            type='upper',
            swap=''),
        3:
        dict(
            name='L1BR',
            id=3,
            color=[255, 0, 0],
            type='upper',
            swap=''),

        4:
        dict(
            name='L2TL',
            id=4,
            color=[255, 100, 0],
            type='upper',
            swap=''),
        5:
        dict(
            name='L2TR',
            id=5,
            color=[255, 100, 0],
            type='upper',
            swap=''),
        6:
        dict(
            name='L2BL',
            id=6,
            color=[255, 100, 0],
            type='upper',
            swap=''),
        7:
        dict(
            name='L2BR',
            id=7,
            color=[255, 100, 0],
            type='upper',
            swap=''),

        8:
        dict(
            name='L3TL',
            id=8,
            color=[53, 238, 151],
            type='upper',
            swap=''),
        9:
        dict(
            name='L3TR',
            id=9,
            color=[53, 238, 151],
            type='upper',
            swap=''),
        10:
        dict(
            name='L3BL',
            id=10,
            color=[53, 238, 151],
            type='upper',
            swap=''),
        11:
        dict(
            name='L3BR',
            id=11,
            color=[53, 238, 151],
            type='upper',
            swap=''),

        12:
        dict(
            name='L4TL',
            id=12,
            color=[53, 121, 222],
            type='upper',
            swap=''),
        13:
        dict(
            name='L4TR',
            id=13,
            color=[53, 121, 222],
            type='upper',
            swap=''),
        14:
        dict(
            name='L4BL',
            id=14,
            color=[53, 121, 222],
            type='upper',
            swap=''),
        15:
        dict(
            name='L4BR',
            id=15,
            color=[53, 121, 222],
            type='upper',
            swap=''),

        16:
        dict(
            name='L5TL',
            id=16,
            color=[0, 255, 255],
            type='upper',
            swap=''),
        17:
        dict(
            name='L5TR',
            id=17,
            color=[0, 255, 255],
            type='upper',
            swap=''),
        18:
        dict(
            name='L5BL',
            id=18,
            color=[0, 255, 255],
            type='upper',
            swap=''),
        19:
        dict(
            name='L5BR',
            id=19,
            color=[0, 255, 255],
            type='upper',
            swap=''),

        20:
        dict(
            name='S1TL',
            id=90,
            color=[153, 0, 153],
            type='upper',
            swap=''),
        21:
        dict(
            name='S1TR',
            id=91,
            color=[153, 0, 153],
            type='upper',
            swap='')
    },
    skeleton_info={
        0:
        dict(link=('L1TL', 'L1TR'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('L1TR', 'L1BR'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('L1BR', 'L1BL'), id=2, color=[0, 255, 0]),
        3:
        dict(link=('L1BL', 'L1TL'), id=3, color=[0, 255, 0]),

        4:
        dict(link=('L2TL', 'L2TR'), id=4, color=[0, 255, 0]),
        5:
        dict(link=('L2TR', 'L2BR'), id=5, color=[0, 255, 0]),
        6:
        dict(link=('L2BR', 'L2BL'), id=6, color=[0, 255, 0]),
        7:
        dict(link=('L2BL', 'L2TL'), id=7, color=[0, 255, 0]),

        8:
        dict(link=('L3TL', 'L3TR'), id=8, color=[0, 255, 0]),
        9:
        dict(link=('L3TR', 'L3BR'), id=9, color=[0, 255, 0]),
        10:
        dict(link=('L3BR', 'L3BL'), id=10, color=[0, 255, 0]),
        11:
        dict(link=('L3BL', 'L3TL'), id=11, color=[0, 255, 0]),

        12:
        dict(link=('L4TL', 'L4TR'), id=12, color=[0, 255, 0]),
        13:
        dict(link=('L4TR', 'L4BR'), id=13, color=[0, 255, 0]),
        14:
        dict(link=('L4BR', 'L4BL'), id=14, color=[0, 255, 0]),
        15:
        dict(link=('L4BL', 'L4TL'), id=15, color=[0, 255, 0]),

        16:
        dict(link=('L5TL', 'L5TR'), id=16, color=[0, 255, 0]),
        17:
        dict(link=('L5TR', 'L5BR'), id=17, color=[0, 255, 0]),
        18:
        dict(link=('L5BR', 'L5BL'), id=18, color=[0, 255, 0]),
        19:
        dict(link=('L5BL', 'L5TL'), id=19, color=[0, 255, 0]),

        20:
        dict(link=('S1TL', 'S1TR'), id=20, color=[0, 255, 0])
        
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1.
    ],
    sigmas=[
        0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025,
        0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025,
        0.025, 0.025
    ])
