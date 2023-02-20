import base64
import io
import smtplib
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import mean_squared_error


def interaction_prod(fset: np.ndarray, inter: int) -> np.ndarray:
    return np.prod(np.array(np.meshgrid(*[fset] * (inter + 1), sparse=True),
                            dtype=object)).flatten().astype(float).tolist()


def create_set(data: pd.DataFrame, target_name: str, back: int, shift: int, interaction: int = 1, flat=True, **kwargs
               ) -> (np.ndarray, np.ndarray):
    """

    :param data: The data in the form of a dataframe you want to convert
    :param target_name: The name of the target/label/...
    :param back: The number of rows to go back
    :param shift: The number of rows the features & targets should be apart
    :param interaction: The number of interaction polynomials
    :param flat: To make each feature line/row a vector (or keep it a matrix)
    :return: features & targets
    """
    features, targets = [], []
    for n in range(back, data.shape[0]-shift+1):
        sub_feat = data.iloc[n-back:n].values
        poly_feat = []
        if interaction > 0:
            for sub in sub_feat:
                poly_feat.append(sub.tolist() + interaction_prod(sub, interaction))
        else:
            poly_feat = sub_feat

        features.append(np.array(poly_feat).flatten().tolist() if flat else np.array(poly_feat).tolist())
        targets.append(data.iloc[n+shift-1:n+shift][target_name].values.flatten()[0])

    return np.array(features, dtype=float), np.array(targets, dtype=float)


def split_train_val_test(features: np.ndarray, targets: np.ndarray, split=(.6, .2, .2), seed=1, **kwargs
                         ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    s_train, s_val, s_test = split
    if sum(split) != 1:
        raise ValueError("split doesn't add up to 1")
    split1 = int(s_train * len(features))
    split2 = int((s_train + s_val) * len(features))
    np.random.seed(seed)
    np.random.shuffle(features)
    np.random.shuffle(targets)
    return (features[:split1], features[split1:split2], features[split2:],
            targets[:split1], targets[split1:split2], targets[split2:])


def sklearn_fit_eval(model,
                     x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray,
                     y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray):
    """"""
    model.fit(x_train, y_train)
    return (mean_squared_error(y_train, model.predict(x_train)),
            mean_squared_error(y_val, model.predict(x_val)),
            mean_squared_error(y_test, model.predict(x_test)))


def flatten_sequence(dictionary, parent_key='', separator='::'):
    """Dynamically reduces a dictionary of depth x to zero depth

    :param dictionary:
    :param parent_key:
    :param separator:
    :return: A dictionary or list depending on the input type. Dictionary if input is dictionary else list
    """
    items = []
    _iterator = dictionary.items() if isinstance(dictionary, dict) else enumerate(dictionary)
    for k, v in _iterator:
        new_key = '%s%s%s' % (parent_key, separator, k) if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_sequence(v, new_key, separator=separator).items())
        elif isinstance(v, (list, set, tuple, np.ndarray)):
            items.extend(flatten_sequence(dict(enumerate(v)), new_key, separator=separator).items())
        else:
            items.append((new_key, v))

    result = dict(items)
    return result if isinstance(dictionary, dict) else list(result.values())


class RWSData:
    """
    https://rijkswaterstaat.github.io/wm-ws-dl/#introduction
    """
    main_path = 'https://waterwebservices.rijkswaterstaat.nl/'
    collect_catalogus = '%sMETADATASERVICES_DBO/OphalenCatalogus/' % main_path
    collect_observations = '%sONLINEWAARNEMINGENSERVICES_DBO/OphalenWaarnemingen' % main_path
    collect_count_observations = '%sONLINEWAARNEMINGENSERVICES_DBO/CheckWaarnemingenAanwezig' % main_path
    collect_latest_observations = '%sONLINEWAARNEMINGENSERVICES_DBO/OphalenLaatsteWaarnemingen' % main_path
    collect_grouped_observations = '%sONLINEWAARNEMINGENSERVICES_DBO/OphalenAantalWaarnemingen' % main_path

    def __init__(self, loc=('ARNH', 700021.921999557, 5762290.374687570),
                 compartment='OW', unit='cm', quantity="WATHTE", characteristic='NAP', group='Dag',
                 fix_data=True):
        self.compartment = compartment
        self.unit = unit
        self.quantity = quantity
        self.characteristic = characteristic
        self.group = group
        self.fix_data = fix_data

        self.code, self.x, self.y = loc

    @staticmethod
    def data_to_str(start, end): return (f"{start.strftime('%Y-%m-%dT%H:%M:%S')}.000+00:00",
                                         f"{end.strftime('%Y-%m-%dT%H:%M:%S')}.000+00:00")

    def meta_descriptions(self):
        """

        :return:
        """
        spec_dict = {"Eenheden": True, "Grootheden": True, "Hoedanigheden": True, "Compartimenten": True, "Parameters": True}
        resp = requests.post(self.collect_catalogus, json={"CatalogusFilter": spec_dict})
        result = resp.json()

        if result['Succesvol']:
            df_meta_list = pd.DataFrame({k: flatten_sequence(v) for k, v in enumerate(result['AquoMetadataLijst'])}).T if 'AquoMetadataLijst' in result.keys() else None
            df_loc_list = pd.DataFrame(result['LocatieLijst']) if 'LocatieLijst' in result.keys() else None
            df_meta_loc_list = pd.DataFrame(result['AquoMetadataLocatieLijst']) if 'AquoMetadataLocatieLijst' in result.keys() else None
            df_height_list = pd.DataFrame(result['BemonsteringshoogteLijst']) if 'BemonsteringshoogteLijst' in result.keys() else None
            df_contractor_list = pd.DataFrame(result['OpdrachtgevendeInstantieLijst']) if 'OpdrachtgevendeInstantieLijst' in result.keys() else None
            df_quality_list = pd.DataFrame(result['KwaliteitswaardecodeLijst']) if 'KwaliteitswaardecodeLijst' in result.keys() else None
            df_ref_list = pd.DataFrame(result['ReferentievlakLijst']) if 'ReferentievlakLijst' in result.keys() else None
            df_status_list = pd.DataFrame(result['StatuswaardeLijst']) if 'LocaStatuswaardeLijsttieLijst' in result.keys() else None

            return (df_meta_list, df_loc_list, df_meta_loc_list, df_height_list, df_contractor_list,
                    df_quality_list, df_ref_list, df_status_list)

    def ddl_data(self):
        catalog_filter = {"Grootheden": True, "Parameters": True, "Compartimenten": True, "Hoedanigheden": True,
                          "Eenheden": True, "BemonsteringsApparaten": True, "BemonsteringsMethoden": True,
                          "BemonsteringsSoorten": True, "BioTaxon": True, "BioTaxon_Compartimenten": True,
                          "MeetApparaten": True, "MonsterBewerkingsMethoden": True, "Organen": True,
                          "PlaatsBepalingsApparaten": True, "Typeringen": True, "WaardeBepalingstechnieken": True,
                          "WaardeBepalingsmethoden": True, "WaardeBewerkingsmethoden": True}

        resp = requests.post(self.collect_catalogus, json={"CatalogusFilter": catalog_filter})
        result = resp.json()
        if result['Succesvol']:
            return (pd.DataFrame({k: flatten_sequence(v) for k, v in enumerate(result['AquoMetadataLijst'])}).T,
                    pd.DataFrame(result['LocatieLijst']),
                    pd.DataFrame(result['AquoMetadataLocatieLijst']),
                    pd.DataFrame(result['BemonsteringshoogteLijst']),
                    pd.DataFrame(result['OpdrachtgevendeInstantieLijst']),
                    pd.DataFrame(result['KwaliteitswaardecodeLijst']),
                    pd.DataFrame(result['ReferentievlakLijst']),
                    pd.DataFrame(result['StatuswaardeLijst']))

    def check_loc_datetime(self, start=datetime.now() - timedelta(days=4), end=datetime.now()):
        start, end = self.data_to_str(start, end)
        request = {
            "AquoMetadataLijst": [{
                "Compartiment": {"Code": self.compartment},
                "Grootheid": {"Code": self.quantity},
                "Eenheid": {"Code": self.unit}
            }],
            "LocatieLijst": [{"X": self.x, "Y": self.y, "Code": self.code}],
            "Periode": {"Begindatumtijd": start, "Einddatumtijd": end}
        }
        resp = requests.post(self.collect_count_observations, json=request)
        true_false_rename = {False: False, True: True, 'False': False, 'True': True, 'false': False, 'true': True}
        return pd.Series(resp.json()).map(true_false_rename)

    def data_loc_datetime(self, start=datetime.now() - timedelta(days=4), end=datetime.now()):
        start, end = self.data_to_str(start, end)
        request = {"AquoPlusWaarnemingMetadata": {"AquoMetadata": {"Compartiment": {"Code": self.compartment},
                                                                   "Grootheid": {"Code": self.quantity},
                                                                   "Eenheid": {"Code": self.unit}}},
                   "Locatie": {"X": self.x, "Y": self.y, "Code": self.code},
                   "Periode": {"Begindatumtijd": start, "Einddatumtijd": end}}
        resp = requests.post(self.collect_observations, json=request)
        result = resp.json()
        return self.get_result(result)

    def data_loc_latest(self):
        request = {"AquoPlusWaarnemingMetadataLijst": [{"AquoMetadata": {"Compartiment": {"Code": self.compartment},
                                                                         "Grootheid": {"Code": self.quantity},
                                                                         "Eenheid": {"Code": self.unit}}}],
                   "LocatieLijst": [{"X": self.x, "Y": self.y, "Code": self.code}]}

        resp = requests.post(self.collect_latest_observations, json=request)
        result = resp.json()
        return self.get_result(result)

    def data_loc_group(self):
        """This has so far no useful purpose"""
        request = {"AquoMetadataLijst": [{"Compartiment": {"Code": self.compartment},
                                          "Grootheid": {"Code": self.quantity},
                                          "Eenheid": {"Code": self.unit}}],
                   "Groeperingsperiode": self.group,
                   "LocatieLijst": [{"X": self.x, "Y": self.y, "Code": self.code}],
                   "Periode": {"Begindatumtijd": self.start, "Einddatumtijd": self.end}}

        resp = requests.post(self.collect_grouped_observations, json=request)
        result = resp.json()
        if result['Succesvol']:
            result_lst = []
            for observation in result['AantalWaarnemingenPerPeriodeLijst']:
                # print(observation.keys())
                # exit()
                result_lst.append([pd.Series(observation['Locatie']),
                                   pd.json_normalize(observation['AantalMetingenPerPeriodeLijst']),
                                   pd.json_normalize(observation['AquoMetadata']).iloc[0]])
            return result_lst
        else:
            warnings.warn(result['Foutmelding'])

    def get_result(self, result):
        if result['Succesvol']:
            result_lst = []
            for observation in result['WaarnemingenLijst']:
                loca = pd.Series(observation['Locatie'])

                meas = pd.json_normalize(observation['MetingenLijst'])
                meas.iloc[:] = [[c[0] if isinstance(c, list) and len(c) == 1 else c for c in r] for r in meas.values]
                meas.set_index('Tijdstip', inplace=True)
                meas.sort_index(inplace=True)
                meas.index = meas.index.str.slice(0, -10).str.replace('T', ' ')
                meas.index = pd.to_datetime(meas.index, format='%Y-%m-%d %H:%M:%S')
                meas = meas.loc[meas['WaarnemingMetadata.KwaliteitswaardecodeLijst'] == '00']
                if self.fix_data:
                    meas['Meetwaarde.Waarde_Numeriek'] = self.fix_spikes(meas['Meetwaarde.Waarde_Numeriek'])

                meta = pd.json_normalize(observation['AquoMetadata']).iloc[0]
                result_lst.append([loca, meas, meta])
            return result_lst
        else:
            warnings.warn(result['Foutmelding'])

    @staticmethod
    def fix_spikes(data: pd.Series):
        def catch_spikes(arr: np.ndarray):
            arr[arr > 2000] = np.nan
            std = np.nanstd(arr)
            d = np.ones(arr.shape, dtype=bool)
            d[:-1] *= np.diff(arr) > 3 * std

            out = np.ones(arr.shape)
            out[d] = np.nan
            out[~d] = 1.
            return out

        def interp_nan(arr: np.ndarray):
            y = np.asarray(list(arr))
            nans, x = np.isnan(y), lambda z: z.nonzero()[0]
            y[nans] = np.interp(x(nans), x(~nans), y[~nans])
            return y

        new_data = data.copy()
        new_data.loc[:] = interp_nan(data.values * catch_spikes(data.values))
        return new_data

    def get_date_range(self, start, end, step):
        dframes = []
        for n in range(0, int(np.ceil((end - start) / step))):
            print(n, start+step*n, start+step*(n+1))
            sub = self.check_loc_datetime(start+step*n, start+step*(n+1))
            if sub['Succesvol'] and sub['WaarnemingenAanwezig']:
                out = self.data_loc_datetime(start+step*n, start+step*(n+1))[0][1]
                dframes.append(out)
        df = pd.concat(dframes)
        df = df.loc[df.index.drop_duplicates()]
        return df


class Emailer:
    def __init__(self, username='noreply.kramer.models@gmail.com', password='KramerNoreply',
                 server='smtp.gmail.com', port=587):
        self.username = username
        self.session = smtplib.SMTP(server, port)
        self.session.ehlo()
        self.session.starttls()
        self.session.ehlo()
        self.session.login(username, password)

    def sendmail(self, recipient='wouter.kramer@me.com', subject='WaterHeights', content=''):
        self.session.sendmail(self.username, recipient,
                              f"From: {self.username}\r\nSubject: {subject}\r\nTo: {recipient}\r\n"
                              f"MIME-Version: 1.0\r\nContent-Type: text/html\r\n\r\n{content}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.quit()


def df_dict_xlsx(df_dict, filepath):
    writer = pd.ExcelWriter(filepath, engine='xlsxwriter')
    for k, v in df_dict.items():
        v.to_excel(writer, sheet_name=k, index=False)
    writer.close()


def fig2html(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    encoded = base64.b64encode(img.read())
    mail_fig = '<img src="data:image/png;base64, {}">'.format(encoded.decode('utf-8'))
    return mail_fig


if __name__ == '__main__':
    # rws = RWSData()
    # mdf = rws.get_date_range(datetime(year=2021, month=1, day=1), datetime.now(), timedelta(days=30))
    # mdf[['Meetwaarde.Waarde_Numeriek']].to_csv('_data/dataset.csv')

    rws = RWSData(compartment='OW', unit='m3/s', quantity="Q")
    mdf = rws.get_date_range(datetime(year=2021, month=1, day=1), datetime.now(), timedelta(days=30))
    mdf[['Meetwaarde.Waarde_Numeriek']].to_csv('_data/dataset_debiet.csv')
