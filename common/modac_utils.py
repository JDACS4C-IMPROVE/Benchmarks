import json
import urllib
import requests
from tqdm import tqdm
import os

<<<<<<< HEAD
=======

modac_user = None
modac_pass = None


>>>>>>> f356751c76b0d5a7b223f901f0a49fb3985de2e6
def get_file_from_modac(fname, origin):
    """ Downloads a file from the "Model and Data Clearning House" (MoDAC)
    repository. Users should already have a MoDAC account to download the data.
    Accounts can be created on modac.cancer.gov

        Parameters
        ----------
        fname : string
            path on disk to save the file
        origin : string
            original MoDAC URL of the file

        Returns
        ----------
        string
            Path to the downloaded file
    """
    print('Downloading data from modac.cancer.gov, make sure you have an account first.')
    total_size_in_bytes = get_dataObject_modac_filesize(origin)
    modac_user, modac_token = authenticate_modac()
    data = json.dumps({})
    headers = {} 
    headers["Content-Type"] = "application/json"
    headers["Authorization"] = "Bearer {0}".format(modac_token)

    post_url = origin + '/download'
    print("Downloading: " + post_url + " ...")
<<<<<<< HEAD
    response = requests.post(post_url, data = data, headers = headers, stream = True)
=======
    response = requests.post(post_url, data=data, headers=headers, auth=auth, stream=True)
>>>>>>> f356751c76b0d5a7b223f901f0a49fb3985de2e6
    if response.status_code != 200:
        print("Error downloading from modac.cancer.gov")
        raise Exception("Response code: {0}, Response message: {1}".format(response.status_code, response.text))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(fname, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        raise Exception("ERROR, something went wrong while downloading ", post_url)

    print('Saved file to: ' + fname)
    return fname


def authenticate_modac(generate_token=False): 
    """
    Authenticates a user on modac.cancer.gov

        Parameters
        ----------
        generate_token : Bool  
            Either generate a new token, or read saved token if it exists 
 
        
        Returns
        ----------
<<<<<<< HEAD
        tuple(string,string) 
            tuple with the (modac_user, modac_token) 
    """
=======
        tuple(string,string)
            tuple with the modac credentials
    """

    global modac_user
    global modac_pass
    if modac_user is None:
        modac_user = input("MoDaC Username: ")
>>>>>>> f356751c76b0d5a7b223f901f0a49fb3985de2e6

    from os.path import expanduser
    home = expanduser("~")
    modac_token_dir =  os.path.abspath(os.path.join(home,".nci-modac"))
    modac_token_file = "credentials.json"
    user_attr = "modac_user"
    token_attr = "modac_token"
    modac_token_path = os.path.join(modac_token_dir, modac_token_file)
    credentials_dic = {}  

    if  not generate_token and os.path.exists(modac_token_path):
        with open(modac_token_path) as f:
          credentials_dic = json.load(f)

    else:
        #Get credentials 
        modac_user = input("MoDaC Username: ")
        import getpass
        modac_pass = getpass.getpass("MoDaC Password: ")

<<<<<<< HEAD
        #Generate token
        auth = (modac_user, modac_pass)
        auth_url = 'https://modac.cancer.gov/api/authenticate'
        print("Authenticating " + modac_user + " ...")
        response = requests.get(auth_url, auth = auth, stream = True)
        if response.status_code != 200:
            print("Error authenticating modac user:{0}", modac_user)
            raise Exception("Response code: {0}, Response message: {1}".format(response.status_code, response.text))

        else: 
            token = response.text
            if not os.path.exists(modac_token_path):
                save_question = "Save MoDaC token in {0}".format(modac_token_path)
                save_token = query_yes_no(save_question)
            else: 
                save_token = True
            if save_token: 
                if not os.path.isdir(modac_token_dir):
                    os.mkdir(modac_token_dir)
                credentials_dic[user_attr] = modac_user 
                credentials_dic[token_attr] =  token
                with open(modac_token_path, "w") as outfile: 
                    json.dump(credentials_dic, outfile, indent=4)
                 
    return (credentials_dic[user_attr], credentials_dic[token_attr])
    
=======
    return (modac_user, modac_pass)

>>>>>>> f356751c76b0d5a7b223f901f0a49fb3985de2e6

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        print(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def get_dataObject_modac_filesize(data_object_path):
    """
    Return the file size in bytes for a modac file
        Parameters
        ----------
        data_object_path : string
            The path of the file on MoDAC

        Returns
        ----------
        integer
            file size in bytes
    """
    self_dic = get_dataObject_modac_meta(data_object_path)
    if "source_file_size" in self_dic.keys():
        return int(self_dic["source_file_size"])
    else:
        return None


def get_dataObject_modac_md5sum(data_object_path):
    """
    Return the md5sum for a modac file
        Parameters
        ----------
        data_object_path : string
            The path of the file on MoDAC

        Returns
        ----------
        string
            The md5sum of the file
    """
    self_dic = get_dataObject_dme_meta(data_object_path)
    if "checksum" in self_dic.keys():
        return self_dic["checksum"]
    else:
        return None


def get_dataObject_modac_meta(data_object_path):
    """
    Return the self metadata values for a file (data_object)
        Parameters
        ----------
        data_object_path : string
            The path of the file on MoDAC

        Returns
        ----------
        dictionary
            Dictonary of all metadata for the file in MoDAC
    """
<<<<<<< HEAD
    #data_object_path = encode_path(data_object_path)
    modac_user, modac_token  = authenticate_modac()
    headers = {} 
    headers["Authorization"] = "Bearer {0}".format(modac_token)
    get_response = requests.get(data_object_path, headers = headers)
=======
    # data_object_path = encode_path(data_object_path)
    auth = authenticate_modac()

    get_response = requests.get(data_object_path, auth=auth)
>>>>>>> f356751c76b0d5a7b223f901f0a49fb3985de2e6
    if get_response.status_code != 200:
        print("Error downloading from modac.cancer.gov", data_object_path)
        raise Exception("Response code: {0}, Response message: {1}".format(get_response.status_code, get_response.text))

    metadata_dic = json.loads(get_response.text)
    self_metadata = metadata_dic['metadataEntries']['selfMetadataEntries']['systemMetadataEntries']
    self_dic = {}
    for pair in self_metadata:
        self_dic[pair['attribute']] = pair['value']

    return self_dic

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--authenticate', action = 'store_true', 
                        help='Authenticate MoDaC user and create token')

    args = parser.parse_args()
    if args.authenticate:
        authenticate_modac(generate_token=True)