import os
import pathlib as pl
from textwrap import dedent

import cls_tables
import cls_tables.data.orf as cls_orf
import pytest

from tests.conftest import (
    DATE_STEP,
    DELTA_1_US,
    FIELD_1,
    FIELD_2,
    FIELD_3,
    INDEX,
    LATITUDE,
    LONGITUDE,
)


@pytest.fixture(scope="session")
def ges_table_dir(tmp_path_factory) -> pl.Path:
    ges_table_dir = tmp_path_factory.mktemp("DSC", numbered=False)
    os.environ["GES_TABLE_DIR"] = str(ges_table_dir)

    return ges_table_dir


@pytest.fixture(scope="session")
def resources_dir(ges_table_dir: pl.Path) -> pl.Path:
    assert ges_table_dir

    rdir = pl.Path(__file__).resolve().parent / "resources"
    os.environ["OCE_DATA"] = str(rdir)

    return rdir


@pytest.fixture(scope="session")
def init_table(tmp_path_factory, ges_table_dir, dataset):
    table_name = "TABLE_TEST"
    table_dir = tmp_path_factory.mktemp(table_name)

    dico_off_file = ges_table_dir / "dictionnaire_parametres_officiel.dsc"
    dico_file = ges_table_dir / "dictionnaire_parametres.dsc"
    dico_cor = ges_table_dir / "dictionnaire_correspondance.txt"
    dico_cor.touch()

    dico_off_content = dedent(
        f"""\
        VERSION  = 0.10

        NOM      = {LATITUDE}
        ID_PARAM = 3
        ID_NOM   = {LATITUDE}
        FMT      = I4
        NB_ITEMS = 1
        UNITE    = degrees_north
        CONV_SI  = 1e-06

        NOM      = {LONGITUDE}
        ID_PARAM = 4
        ID_NOM   = {LONGITUDE}
        FMT      = I4
        NB_ITEMS = 1
        UNITE    = degrees_east
        CONV_SI  = 1e-06

        NOM      = {FIELD_1}
        ID_PARAM = 5
        ID_NOM   = {FIELD_1}
        FMT      = I4
        NB_ITEMS = 1
        UNITE    = dB
        CONV_SI  = 1

        NOM      = {FIELD_2}
        ID_PARAM = 6
        ID_NOM   = {FIELD_2}
        FMT      = R4
        NB_ITEMS = 1
        UNITE    = m
        CONV_SI  = 1

        NOM      = {FIELD_3}
        ID_PARAM = 7
        ID_NOM   = {FIELD_3}
        FMT      = I4
        NB_ITEMS = 1
        UNITE    = m
        CONV_SI  = 1
        """
    )
    dico_off_file.write_text(dico_off_content)
    dico_file.write_text('#include "dictionnaire_parametres_officiel.dsc"\n')

    dsc_file = ges_table_dir / f"{table_name}.dsc"
    dsc_content = dedent(
        f"""\
        REPERT_TABLE    = {table_dir}
        DELTA_T         = 1.0
        POURCENT_LIST   = 5
        TOLERANCE_DT   = 0.0
        FREQUENCE_STOCKAGE = 24

        ID_PARAM     = 3
        ID_PARAM     = 4
        ID_PARAM     = 5
        ID_PARAM     = 6
        ID_PARAM     = 7
        """
    )
    dsc_file.write_text(dsc_content)

    table = cls_tables.TableMeasure(table_name, mode="w")
    table.write_values_as_dataset(dataset)
    table.close()

    return table_name


@pytest.fixture(scope="session")
def table_name(init_table) -> str:
    return init_table


@pytest.fixture(scope="session")
def init_orf(resources_dir, dataset):
    assert resources_dir

    orf_name = "ORF_NAME"
    time_values = dataset[INDEX].values
    orf = cls_orf.Orf.create(orf_name, "C2")
    track = cls_orf.OrfTrack(
        cls_orf.OrfPosition(
            cls_tables.round_vanilla_datetime(date=time_values[0]), 10, -60
        ),
        cls_orf.OrfPosition(
            cls_tables.round_vanilla_datetime(date=time_values[0]), 5, None
        ),
        cls_orf.OrfPosition(
            cls_tables.round_vanilla_datetime(date=time_values[2] - DELTA_1_US), 20, 60
        ),
        1,
        1,
    )
    orf.write_track(track)

    track = cls_orf.OrfTrack(
        cls_orf.OrfPosition(
            cls_tables.round_vanilla_datetime(date=time_values[2]), 10, -60
        ),
        cls_orf.OrfPosition(
            cls_tables.round_vanilla_datetime(date=time_values[2]), 5, None
        ),
        cls_orf.OrfPosition(
            cls_tables.round_vanilla_datetime(date=time_values[3] - DELTA_1_US), 20, 60
        ),
        1,
        2,
    )
    orf.write_track(track)

    track = cls_orf.OrfTrack(
        cls_orf.OrfPosition(
            cls_tables.round_vanilla_datetime(date=time_values[3]), 10, -60
        ),
        cls_orf.OrfPosition(
            cls_tables.round_vanilla_datetime(date=time_values[3]), 5, None
        ),
        cls_orf.OrfPosition(
            cls_tables.round_vanilla_datetime(date=time_values[4] - DELTA_1_US), 20, 60
        ),
        3,
        1,
    )
    orf.write_track(track)

    track = cls_orf.OrfTrack(
        cls_orf.OrfPosition(
            cls_tables.round_vanilla_datetime(date=time_values[4]), 10, -60
        ),
        cls_orf.OrfPosition(
            cls_tables.round_vanilla_datetime(date=time_values[4]), 5, None
        ),
        cls_orf.OrfPosition(
            cls_tables.round_vanilla_datetime(date=time_values[4] + DATE_STEP), 20, 60
        ),
        3,
        3,
    )
    orf.write_track(track)

    orf.close()

    return orf_name


@pytest.fixture(scope="session")
def orf_name(init_orf) -> str:
    return init_orf
