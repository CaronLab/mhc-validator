import pandas as pd
import xmltodict
from pathlib import Path
from os import PathLike
from typing import Union, List
from collections import OrderedDict
from copy import deepcopy


def find_not_common_items(*sets):
    if len(sets) < 2:
        return set()
    common = sets[0] & sets[1]
    uncommon = set()
    for s in sets:
        common &= s
    for s in sets:
        uncommon.update(s - common)
    return uncommon


def pepxml_to_mhcv(pepxml_file: Union[str, PathLike],
                   destination: Union[str, PathLike] = None,
                   decoy_prefix: str = 'rev_',
                   split_output: bool = False):

    pepxml_file = Path(pepxml_file)

    if destination is None:
        destination = pepxml_file.parent
    else:
        destination = Path(destination)

    if destination.is_dir():
        destination = destination / f'{pepxml_file.stem.split(".")[0]}.mhcv'

    print(f'Loading {pepxml_file.name}')
    pepxml_dict = pepxml_to_dict(pepxml_file=pepxml_file)

    print('Converting to MHCV format')
    mhcv_files = pepxml_dict_to_mhcv(pepxml_dict=pepxml_dict,
                                     destination=destination,
                                     decoy_prefix=decoy_prefix,
                                     split_output=split_output)
    return mhcv_files


def pepxml_to_dict(pepxml_file: Union[str, PathLike]) -> dict:
    """
    Parse a pepXML file and return a dictionary. Is only a thin wrapper around xmltodict.parse.
    :param pepxml_file: Path to the pepXML file.
    :return: a dictionary containing the pepXML data.
    """

    with open(pepxml_file, 'r') as f:
        data = f.read()

    pepxml_dict = xmltodict.parse(data)
    return pepxml_dict


def pepxml_dict_to_mhcv(pepxml_dict: Union[dict, OrderedDict], destination: Union[str, PathLike],
                        decoy_prefix: str = 'decoy_', split_output: bool = False):
    """
    Parse a dictionary returned by pepxml_to_dict and write a MHCV file compatible with MhcValidator (and Percolator).
    :param pepxml_dict: The dictionary returned by a call to pepxml_to_dict
    :return: None
    """

    destination: Path = Path(destination)

    msms_run: Union[OrderedDict, List[OrderedDict]] = deepcopy(pepxml_dict['msms_pipeline_analysis']['msms_run_summary'])

    if isinstance(msms_run, OrderedDict):
        msms_run: List[OrderedDict] = [msms_run]

    ms_files = set()
    search_engines = set()
    search_engine_scores = set()
    for run in msms_run:
        ms_files.update([Path(run['@base_name']).stem.split('.')[0]])
        search_engines.update([run['search_summary']['@search_engine']])

        i = 0
        while i < len(run['spectrum_query']):
            if run['spectrum_query'][i]['search_result'] is None:
                i += 1
                continue
            elif 'search_hit' not in run['spectrum_query'][i]['search_result'].keys():
                i += 1
                continue
            scores = run['spectrum_query'][i]['search_result']['search_hit']['search_score']
            if isinstance(scores, OrderedDict):
                scores = [scores]
            for score in scores:
                search_engine_scores.update([run['search_summary']['@search_engine'] + '_' + score['@name']])
            break
    search_engine_scores = list(search_engine_scores)
    search_engine_scores.sort()

    # get possible charge states
    min_charge = 1e6
    max_charge = -1e6
    for run in msms_run:
        if 'spectrum_query' in run.keys():
            for spectrum in run['spectrum_query']:
                charge = int(spectrum['@assumed_charge'])
                if charge < min_charge:
                    min_charge = charge
                elif charge > max_charge:
                    max_charge = charge

    outputs = {}
    headers = {}
    for run in msms_run:
        if 'spectrum_query' not in run.keys():  # there is a msms_run for each search engine and MS file, but only one spectrum_query per MS file.
            continue

        if split_output:
            header_start = ['SpecId', 'Label', 'ScanNr', 'ExpMass', 'CalcMass', 'MassDiff']
        else:
            header_start = ['SpecId', 'Label', 'ScanNr', *ms_files, *search_engines, 'ExpMass', 'CalcMass', 'MassDiff',
                            *search_engine_scores]

        header_end = ['Peptide', 'Proteins']

        search_engine = run['search_summary']['@search_engine']
        ms_file = Path(run['search_summary']['@base_name']).stem.split('.')[0]

        scans = OrderedDict()

        for spectrum in run['spectrum_query']:

            if spectrum['search_result'] is None:
                continue
            elif 'search_hit' not in spectrum['search_result'].keys():
                continue

            if isinstance(spectrum['search_result']['search_hit'], list):
                search_list = spectrum['search_result']['search_hit']
                search_list.sort(key=lambda x: int(x['@hit_rank']))
                if search_list[0]['@hit_rank'] == search_list[1]['@hit_rank']:
                    print(f"{spectrum['@spectrum']} {search_engine} "
                          f"search has more than one top-ranking hit. Selecting the first.")
                spectrum['search_result']['search_hit'] = search_list[0]

            scan = dict()
            scan['ScanNr'] = spectrum['@start_scan']
            scan['SpecId'] = f'{ms_file}_scan={scan["ScanNr"]}'

            if not split_output:
                for msf in ms_files:
                    if ms_file == msf:
                        scan[msf] = '1'
                    else:
                        scan[msf] = '0'
                for se in search_engines:
                    if search_engine == se:
                        scan[se] = '1'
                    else:
                        scan[se] = '0'

            # one-hot encode the charge
            charge = int(spectrum['@assumed_charge'])
            for i in range(min_charge, max_charge+1):
                if charge == i:
                    scan[f'Charge_{i}'] = '1'
                else:
                    scan[f'Charge_{i}'] = '0'

            scan['ExpMass'] = spectrum['@precursor_neutral_mass']
            scan['CalcMass'] = spectrum['search_result']['search_hit']['@calc_neutral_pep_mass']
            scan['MassDiff'] = spectrum['search_result']['search_hit']['@massdiff']
            #exp_mass = float(scan['ExpMass'])
            #mass_diff = float(scan['MassDiff'])
            #scan['ppmMassDiff'] = str(round((mass_diff / exp_mass) * 1e6, 4))

            '''if '@tot_num_ions' in spectrum['search_result']['search_hit'].keys():
                scan['Ions'] = spectrum['search_result']['search_hit']['@tot_num_ions']
                scan['MatchedIons'] = spectrum['search_result']['search_hit']['@num_matched_ions']
                scan['MatchedIonFrac'] = str(round(float(scan['MatchedIons']) / float(scan['Ions']), 4))'''

            proteins = [spectrum['search_result']['search_hit']['@protein']]
            if 'alternative_protein' in spectrum['search_result']['search_hit'].keys():
                alt_prots = spectrum['search_result']['search_hit']['alternative_protein']
                if isinstance(alt_prots, OrderedDict):
                    alt_prots = [alt_prots]
                for prot in alt_prots:
                    proteins.append(prot['@protein'])
            scan['Proteins'] = '\t'.join(proteins)

            is_target = False
            for prot in proteins:
                if not prot.startswith(decoy_prefix):
                    is_target = True
            scan['Label'] = '1' if is_target else '-1'

            prev_aa = spectrum['search_result']['search_hit']['@peptide_prev_aa']
            next_aa = spectrum['search_result']['search_hit']['@peptide_next_aa']

            if 'modification_info' in spectrum['search_result']['search_hit'].keys():
                mods = spectrum['search_result']['search_hit']['modification_info'].get('mod_aminoacid_mass', [])
                nterm_mod = spectrum['search_result']['search_hit']['modification_info'].get('@mod_nterm_mass', False)
                cterm_mod = spectrum['search_result']['search_hit']['modification_info'].get('@mod_cterm_mass', False)
                peptide: str = spectrum['search_result']['search_hit']['@peptide']
                if isinstance(mods, OrderedDict):
                    mods = [mods]
                modifications = []
                for mod in mods:
                    modifications.append((int(mod['@position']), mod['@mass']))
                modifications.sort(key=lambda x: x[0], reverse=True)
                for mod in modifications:
                    peptide = peptide[:mod[0]] + f'[{mod[1]}]' + peptide[mod[0]:]
                if nterm_mod:
                    peptide = f'n[{nterm_mod}]{peptide}'
                if cterm_mod:
                    peptide = f'{peptide}n[{cterm_mod}]'
                scan['Peptide'] = f"{prev_aa}.{peptide}.{next_aa}"
            else:
                scan['Peptide'] = f"{prev_aa}.{spectrum['search_result']['search_hit']['@peptide']}.{next_aa}"

            if not split_output:
                for score in search_engine_scores:
                    scan[score] = '0'

            scores = spectrum['search_result']['search_hit']['search_score']
            if isinstance(scores, OrderedDict):
                scores = [scores]
            for score in scores:
                scan[run['search_summary']['@search_engine'] + '_' + score['@name']] = score['@value']

            if 'analysis_result' in spectrum['search_result']['search_hit'].keys():
                analysis_result = spectrum['search_result']['search_hit']['analysis_result']
                if isinstance(analysis_result, OrderedDict):
                    analysis_result = [analysis_result]
                for result in analysis_result:
                    analysis = result['@analysis']
                    scan[f'{analysis}_probability'] = result[f'{analysis}_result']['@probability']
                    if '@all_ntt_prob' in result[f'{analysis}_result'].keys():
                        ntt_probs = result[f'{analysis}_result']['@all_ntt_prob'].replace('(', '')\
                            .replace(')', '').split(',')
                        for i, ntt_prob in enumerate(ntt_probs):
                            scan[f'{analysis}_ntt_prob_{i}'] = ntt_prob
                    for parameter in result[f'{analysis}_result']['search_score_summary']['parameter']:
                        scan[f'{analysis}_{parameter["@name"]}'] = parameter["@value"]

            scans[scan['SpecId']] = scan

        header_middle = []
        for key in next(iter(scans.values())).keys():
            if key not in header_start and key not in header_end:
                header_middle.append(key)
        outputs[f'{ms_file}_{search_engine}'] = scans
        scans = OrderedDict()
        headers[f'{ms_file}_{search_engine}'] = header_start + header_middle + header_end

    if not split_output:
        for header in headers.values():
            if header != next(iter(headers.values())):
                raise IndexError('split_output was set to False, but the headers of each run are not the same. The '
                                 f'following elements are not found in all runs: '
                                 f'{find_not_common_items([set(x) for x in headers.values()])}')

        header = next(iter(headers.values()))

        with open(destination, 'w') as f:
            f.write('\t'.join(next(iter(headers.values()))) + '\n')
            for run, scans in outputs.items():
                for spectrum in scans.values():
                    f.write('\t'.join([spectrum[h] for h in header]) + '\n')

        return [destination]

    else:
        fouts = []
        for run, scans in outputs.items():
            header = headers[run]
            fout = destination.parent / f'{destination.stem.split(".")[0]}_{run}.mhcv'
            with open(fout, 'w') as f:
                f.write('\t'.join(header) + '\n')
                for spectrum in scans.values():
                    f.write('\t'.join([spectrum[h] for h in header]) + '\n')
            fouts.append(fout)

        return fouts
