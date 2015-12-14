# in/out functions for vm_tools
# import h5py
import warnings
import numpy as np
from scipy.io import loadmat,savemat
# Local stuff
# from .Classes.Stimulus import Stimulus #, Labels? (subclass Stimulus to allow for non-image/auditory/whatever inputs?)
# from .Classes.FeatureSpace import FeatureSpace # Change to: FeatureSpace
# from .Classes.Mask import Mask
# from .Classes.ROI import ROISet,ROIMask
# from .Classes.DataSet import fMRI_DataSet 
# from .Classes.Model import fMRI_Model 
# from .options import config
# from .utils import as_list
# from .dbwrapper import docdbs
# is_verbose = config.get('general','is_verbose').lower() in ('true','yes','t','y','1') # global pkg verbosity option?
is_verbose = True

# add generic load_variable function, agnostic to data type?


# Move to .io module or discard...
# def h5str(fName,node):
#     '''
#     Pull a string variable from an hf5 file
#     '''
#     T = tables.openFile(fName)
#     ss = T.getNode('/'+node)[:].flatten()
#     T.close()
#     strOut = ''
#     for s in ss:
#         strOut+=chr(s)
#     return strOut
def _query(dbi,kw,docclass):
    """Helper query function that avoids stupid caching issues"""
    doc = dbi.query_documents(1,**kw)
    return docclass.from_docdict(doc,dbi)

def get_Stimulus(dbi,stim_q):
    """Get a SINGLE stimulus from database, given partial match. 

    All stimuli are returned as lists, since a single stimulus can
    have multiple parts, except multi-component stimuli (in which case
    each component is returned as a list). This function assures that
    all results are parts of the same stimulus. 

    Parameters
    ----------
    dbi : DocDBInterface
        couchdb interface class
    stim_q : 
    
    Notes
    -----
    Don't specify return_objects in your arguments; that's implied (this function always returns objects)
    """
    if ('stim_class' in stim_q) and (stim_q['stim_class']=='multi_component'):
        S = {}
        for k,v in stim_q.items():
            if v=='multi_component':
                S[k] = v
                continue
            # All of these may be multi-part; they don't necessarily have to have
            # the same number of parts... (do they?)
            S[k] = get_Stimulus(dbi,v)
        return S
    else:
        S = dbi.query_documents(**stim_q)
        if not S[0]['n_parts']==len(S):
            raise Exception('Number of parts in stimulus do not add up!')
        S = sorted(S,key=lambda x:x['part'])
    return [Stimulus.from_docdict(s,dbi) for s in S]

def get_FeatureSpace(dbi,stim_q,fs_q,is_multipart=False):
    """Query for a SINGLE feature space. 

    (Slightly) modified docdb query. Always searches for a single FeatureSpace
    object. If your query returns multiple objects, an error is thrown.

    Parameters
    ----------
    stim_q : dict 
        named arguments to query for stimulus (always queries for oStimulus)
    fs_q : dict 
        named arguments to query for feature space
    is_multipart : bool
        Whether the FeatureSpace in question has multiple parts (e.g., multiple
        sessions).  If true, returns a list. Otherwise, returns a single 
        vm_tools.FeatureSpace instance.

    
    Returns
    -------
    FS : FeatureSpace object 
        Single FeatureSpace object that matches query

    Example
    -------
    trn_fs = get_featureSpace(stim_q=dict(exp='awesome',session=1,stim_class='movie',trnval='trn'), 
                                fs_q=dict(ppseq=['preprocColorSpace',[1],'preprocWavelets_grid',[1]]))

    Notes
    -----
    is_joint_fs : bool
        Whether to query for a joint feature space (multiple feature spaces, all
        meant to be fit together, returned as a dictionary or - if `is_multipart` -
        as a list of dictionaries)

    Don't provide "type" arguments to stim_q and fs_q; they are provided in the code.
    Don't specify return_objects in your arguments; that's implied (this function always returns objects)
    """
    is_joint_fs = isinstance(fs_q,dict) and ('joint' in fs_q.keys())
    if is_joint_fs:
        # Advanced options; all keys should be the same in stim_q and fs_q
        FS = {}
        for k in fs_q.keys():
            if k=='joint': continue
            FS[k] = get_FeatureSpace(dbi,stim_q[k],fs_q[k],is_multipart=is_multipart)
            if len(fs_q.keys())==2:
                # Special case: only one feature space {"A":blah,"joint":True} - treat this like any single feature space
                FS = FS[k]
                break
    else:
        if is_verbose:
            print('Stimulus query:')
        S = get_Stimulus(dbi,stim_q)
        if isinstance(S,dict):
            # Multi-component stimulus
            oStimulus = {}
            for k,v in S.items():
                if k=='stim_class':
                    oStimulus[k] = v
                else:
                    oStimulus[k] = [vv._id for vv in v]
                    # Already sorted! ... right???
                    # = sorted(tmpS,key=lambda x: x.part)
            if is_multipart:
                # Split up different parts of each component
                n_parts = [len(v) for v in oStimulus.values()]
                if not all([n==n_parts[0] for n in n_parts]):
                    raise Exception("Cannot split multi-component stimulus into multiple parts; different components have different numbers of parts!")
                else:
                    n_parts = n_parts[0]
                oStimulus = [dict((k,[v[ii]]) for k,v in oStimulus.items()) for ii in range(n_parts)]
        else:
            oStimulus = [s._id for s in S]
        if is_verbose:
            print('FeatureSpace query:')
        if is_multipart:
            # Each part of the stimulus 
            FS = [dbi.query_documents(1,type='FeatureSpace',oStimulus=s,**fs_q) for s in oStimulus]
            FS = [FeatureSpace.from_docdict(fs,dbi) for fs in FS]
        else:
            FS = dbi.query_documents(1,type='FeatureSpace',oStimulus=oStimulus,**fs_q)
            FS = FeatureSpace.from_docdict(FS,dbi)
    return FS

def get_fs_matrix(fs_list,data=None,repeats2timecourse=False):
    """Creates composite feature space matrix from potentially numerous components

    Requires data object input (don't load it, though!) if un-raveling feature space to 
    predict re-scrambled data (see help in DataSet object(s))

    """
    X = []
    if data is None:
        data = range(len(fs_list))
    for d,fs in zip(data,fs_list):
        if isinstance(fs,dict):
            # Composite feature space - stack horizontally
            # Note: always concatenate models in alphabetical order of dict keys, for regularity
            if repeats2timecourse:
                tmp = np.hstack([d.repeats2timecourse(fs[k].load(in_place=False)) for k in sorted(fs.keys())])
            else:
                tmp = np.hstack([fs[k].load(in_place=False) for k in sorted(fs.keys())])
        else:
            # Single feature space - just load 
            if repeats2timecourse:
                tmp = d.repeats2timecourse(fs.load(in_place=False))
            else:
                tmp = fs.load(in_place=False)
        X.append(tmp)
    return np.vstack(X)

def get_Mask(dbi,mask_q):
    """Get and load a SINGLE mask from a database 

    ... Or a combined mask, which is technically multiple masks. Wrapper for
    docdb query (/queries)

    Parmeters
    ---------
    dbi : docdb.DocDBInterface
        interface for database in which masks are stored
    mask_q : dict
        keyword arguments for query
    Notes
    -----
    Don't specify return_objects in mask_q; that's implied (this function always returns objects)
    """
    # This is under-explored code: I've still not fit any double-masked data...
    if isinstance(mask_q,(list,tuple)):
        return [_query(dbi,mq,Mask) for mq in mask_q]
    else:
        mask_q.update(dict(type='Mask'))
        return _query(dbi,mask_q,Mask)

def get_ROISet(dbi,roi_q):
    """Get and load an ROISet object from a database.

    Parameters
    ----------
    dbi : docdb.DocDBInterface
        Interface for database in which ROISet is stored
    roi_q : dict
        keyword arguments for query
    
    Notes
    -----
    Don't specify return_objects in roi_q; that's implied (this function always returns objects)
    """
    if isinstance(roi_q,(list,tuple)):
        return [_query(dbi,mq,ROISet) for mq in roi_q]
    else:
        roi_q.update(dict(type='ROISet'))
        return _query(dbi,roi_q,ROISet)


def get_fMRI_DataSet(dbi,ds_q,mask_q=None):
    '''Get a SINGLE fMRI_DataSet from database

    Wrapper for docdb query.

    Parameters
    ---------- 
    dbi : (docdb.DocDBInterface instance)
        database interface object from docdb
    mask_q : dict or None

    mask_arg : tuple
        All `*args` and `**kwargs` will be fed to dbi.query(*args,**kwargs)

    Other Inputs
    ------------

    Returns
    -------
    
    Notes
    -----

    '''
    if mask_q is None:
        mask = 'None'
    else:
        mask = _query(dbi,mask_q,Mask)
        mask = mask._id
    DS = dbi.query_documents(1,type='fMRI_DataSet',mask=mask,**ds_q)
    DS = fMRI_DataSet.from_docdict(DS,dbi)
    return DS

def get_fMRI_Model(dbi,mod_q,trn_data_q,trn_fs_q,trn_stim_q,mask_q=None,pre_mask_q=None,
                        val_data_q=None,val_fs_q=None,val_stim_q=None):
    # Get trn/val data
    trn_data = [get_fMRI_DataSet(dbi,ds_q=tdq,mask_q=pre_mask_q) for tdq in as_list(trn_data_q)]
    if not val_data_q is None:
        val_data = [get_fMRI_DataSet(dbi,ds_q=vdq,mask_q=pre_mask_q) for vdq in as_list(val_data_q)]
    else:
        val_data = None
    # Get trn/val feature spaces
    trn_fs = [get_FeatureSpace(dbi,fs_q=tfq,stim_q=tsq) for tfq,tsq in zip(as_list(trn_fs_q),as_list(trn_stim_q))]
    if not val_fs_q is None:
        val_fs = [get_FeatureSpace(dbi,fs_q=vfq,stim_q=vsq) for vfq,vsq in zip(as_list(val_fs_q),as_list(val_stim_q))]
    # Get mask
    mask,pre_mask = mask_q,pre_mask_q
    if not mask is None:
        mask = get_Mask(dbi,mask_q=mask)
    # Potentially combine masks
    if pre_mask is None:
        cmask = mask
    else:
        pre_mask = get_Mask(dbi,mask_q=pre_mask)
        if mask is None:
            cmask = pre_mask
        else:
            print('WARNING! mask combination code probably needs more testing!')
            cmask = [pre_mask, mask]
    # if mask is None and pre_mask is None:
    #     cmask = None
    # Get model
    if val_data is None:
        tmp = fMRI_Model(trn_data=trn_data,trn_fs=trn_fs,
                    mask=cmask,**mod_q)
        mod = dbi.query_documents(1,**tmp.docdict)
        mod = fMRI_Model.from_docdict(mod,dbi)
    else:
        tmp = fMRI_Model(trn_data=trn_data,trn_fs=trn_fs,
                    val_data=val_data,val_fs=val_fs,mask=cmask,
                    **mod_q)
        mod = dbi.query_documents(1,**tmp.docdict)
        mod = fMRI_Model.from_docdict(mod,dbi)
    return mod

def get_fMRI_Simulation(*args,**kwargs):
    """
    Same as "GetSims", but returns a single STRFsim instead of a list. 
    Throws an error if the keyword arguments are not specific enough / if
    no models that fit the key/value pairs.
    """
    M = GetSims(*args,**kwargs)
    if len(M) < 1:
        raise Exception("No STRFsims fit the query terms used! Try again!")
    elif len(M)==1:
        return M[0]
    elif len(M)>1:
        raise Exception("Multiple STRFsims fit the query terms used! Please be more specific!")




def GetConfMat(*args,**kwargs):
    warnings.warn('Deprecated! Use get_Decode')
    return get_Decode(*args,**kwargs)

def get_Decode(sub_id=None,task='ObjCount',VoxSel=None,
        type='fMRI_Decode',DecodeType='Dist-Euclidean',**kwargs):
    '''
    Retrieve fit models from STRFdb

    kwargs are all the query terms you wish to use for searching the database. Examples below.

    Returns a list of models matching query terms.

    Examples:
    ModPart='HoG' # Retrieves all models with "HoG" in their model name string ("Mod" field in the database) for subject ML

    '''
    raise NotImplementedError("i is broke as shit!")
    import ml_Utils as ml # This is shit get rid of me
    import numpy as np
    inpt = locals()
    query = {}
    for key,val in inpt.items():
        if key in ['kwargs']:
            continue
        if not val is None:
            query[key] = val
    for key,val in kwargs.items():
        if not val is None:
            query[key] = val
    db = docdb.getclient(dbname='strfdb-ml',queries=('basic','models'),return_objects=False,is_verbose=is_verbose)
    ModelDicts = db.query(**query)
    STRFmodels = []
    for m in ModelDicts:
        hf = h5py.File(m['Path'])
        ConfMat = np.array(hf['/ConfMat'])
        m['ConfMat'] = ConfMat
    ConfMats = [ml.mlDict(m) for m in ModelDicts]
    return ConfMats