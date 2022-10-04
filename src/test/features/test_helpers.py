import os
from pathlib import Path

import numpy as np
import pandas as pd
import torchvision.transforms.functional as transform

from src.db import run_query
from src.features.image_variations import numpy_to_tensor


def get_sample_images_directory():
    currdir = Path(__file__).resolve()
    return currdir.parent / "sample_images_small"


def get_sample_images_blocklots():
    return os.listdir(get_sample_images_directory())


def get_sample_image(blocklot, grayscale):
    blocklot_fpath = get_sample_images_directory() / blocklot
    image = numpy_to_tensor(np.load(blocklot_fpath))
    if grayscale:
        return transform.rgb_to_grayscale(image)
    return image


def get_sample_images(blocklots, grayscale):
    sample_images = []
    for blocklot in blocklots:
        sample_images.append(get_sample_image(blocklot, grayscale))
    return sample_images


def get_sample_blocklots():
    b1 = "1451 024"
    b2 = "0193 053"
    b3 = "2406 003"
    b4 = "1504 071"
    b5 = "4605 A011"
    b6 = "0109 001"
    b7 = "0065 001"
    b8 = "0122 060"
    b9 = "0005 011"
    b10 = "0452 026"
    return [b1, b2, b3, b4, b5, b6, b7, b8, b9, b10]


def run_sample_query(blocklots):

    data = run_query(
        """
        SELECT tp.blocklot, 
            -- Limited to attributes that are particularly interesting or not empty 
            roof.mainrooffe as roof_damage, mainroofpc as roof_damage_score,
            groupstatu AS demo_status, decision AS demo_decision, 
            iia.name AS iia_name, 
            onv.datenotice AS vacancy_notice, onv.noticetype 
        FROM raw.tax_parcel_address AS tp
        LEFT JOIN raw.building_construction_permits AS bcp
            ON tp.blocklot = bcp.blocklot
        LEFT JOIN raw.communitydevelopmentzones AS cdz
            ON ST_Intersects(tp.wkb_geometry, cdz.shape)
        LEFT JOIN raw.completedcitydemolition AS ccd
            ON ST_Intersects(tp.wkb_geometry, ccd.shape)
        LEFT JOIN raw.developmentdivisionprojects AS ddp
            ON ST_Intersects(tp.wkb_geometry, ddp.shape)
        LEFT JOIN raw.filedandopenreceiverships AS r
            ON tp.blocklot = r.blocklot   
        LEFT JOIN raw.impactinvestmentareas AS iia
            ON ST_Intersects(tp.wkb_geometry, iia.shape)
        LEFT JOIN raw.majorredevelopmentareas AS mra
        ON ST_Intersects(tp.wkb_geometry, mra.shape)
        LEFT JOIN raw.open_notice_vacant AS onv
            ON ST_Intersects(tp.wkb_geometry, onv.wkb_geometry)
        LEFT JOIN raw.real_estate_data AS red 
            ON tp.blocklot = red.blocklot 
        LEFT JOIN raw.roofdata_2018 AS roof
            ON tp.blocklot = roof.blocklot
        WHERE tp.blocklot = ANY(%s)
        ORDER BY roof_damage_score
    """,
        (blocklots,),
    )
    df = pd.DataFrame(data, columns=data[0].keys())
    # Add column and sample image filenames
    df["image_id"] = ["image" + str(i) for i in range(df.shape[0])]
    return df
