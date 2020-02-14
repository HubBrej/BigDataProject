import base64
import logging
import boto3
from botocore.exceptions import ClientError
from cryptography.fernet import Fernet

def retrieve_cmk(desc):
    """Retrieve an existing KMS CMK based on its description
    :param desc: Description of CMK specified when the CMK was created
    :return Tuple(KeyId, KeyArn) where:
        KeyId: CMK ID
        KeyArn: Amazon Resource Name of CMK
    :return Tuple(None, None) if a CMK with the specified description was
    not found
    """
    kms_client = boto3.client('kms')
    try:
        response = kms_client.list_keys()
    except ClientError as e:
        logging.error(e)
        return None, None

    done = False
    while not done:
        for cmk in response['Keys']:
            try:
                key_info = kms_client.describe_key(KeyId=cmk['KeyArn'])
            except ClientError as e:
                logging.error(e)
                return None, None

            if key_info['KeyMetadata']['Description'] == desc:
                return cmk['KeyId'], cmk['KeyArn']

        if not response['Truncated']:
            logging.debug('A CMK with the specified description was not found')
            done = True
        else:
            try:
                response = kms_client.list_keys(Marker=response['NextMarker'])
            except ClientError as e:
                logging.error(e)
                return None, None

    return None, None


def create_cmk(desc='Customer Master Key'):
    """Create a KMS Customer Master Key
    The created CMK is a Customer-managed key stored in AWS KMS.
    :param desc: key description
    :return Tuple(KeyId, KeyArn) where:
        KeyId: AWS globally-unique string ID
        KeyArn: Amazon Resource Name of the CMK
    :return Tuple(None, None) if error
    """

    kms_client = boto3.client('kms')
    try:
        response = kms_client.create_key(Description=desc)
    except ClientError as e:
        logging.error(e)
        return None, None

    return response['KeyMetadata']['KeyId'], response['KeyMetadata']['Arn']


def create_data_key(cmk_id, key_spec='AES_256'):
    """Generate a data key to use when encrypting and decrypting data
    :param cmk_id: KMS CMK ID or ARN under which to generate and encrypt the
    data key.
    :param key_spec: Length of the data encryption key. Supported values:
        'AES_128': Generate a 128-bit symmetric key
        'AES_256': Generate a 256-bit symmetric key
    :return Tuple(EncryptedDataKey, PlaintextDataKey) where:
        EncryptedDataKey: Encrypted CiphertextBlob data key as binary string
        PlaintextDataKey: Plaintext base64-encoded data key as binary string
    :return Tuple(None, None) if error
    """

    kms_client = boto3.client('kms')
    try:
        response = kms_client.generate_data_key(KeyId=cmk_id, KeySpec=key_spec)
    except ClientError as e:
        logging.error(e)
        return None, None

    return response['CiphertextBlob'], base64.b64encode(response['Plaintext'])



def decrypt_data_key(data_key_encrypted):
    """Decrypt an encrypted data key
    :param data_key_encrypted: Encrypted ciphertext data key.
    :return Plaintext base64-encoded binary data key as binary string
    :return None if error
    """

    kms_client = boto3.client('kms')
    try:
        response = kms_client.decrypt(CiphertextBlob=data_key_encrypted)
    except ClientError as e:
        logging.error(e)
        return None

    return base64.b64encode((response['Plaintext']))



NUM_BYTES_FOR_LEN = 4

def encrypt_file(filename, cmk_id):
    """Encrypt a file using an AWS KMS CMK
    A data key is generated and associated with the CMK.
    The encrypted data key is saved with the encrypted file. This enables the
    file to be decrypted at any time in the future and by any program that
    has the credentials to decrypt the data key.
    The encrypted file is saved to <filename>.encrypted
    Limitation: The contents of filename must fit in memory.
    :param filename: File to encrypt
    :param cmk_id: AWS KMS CMK ID or ARN
    :return: True if file was encrypted. Otherwise, False.
    """
    try:
        with open(filename, 'rb') as file:
            file_contents = file.read()
    except IOError as e:
        logging.error(e)
        return False

    data_key_encrypted, data_key_plaintext = create_data_key(cmk_id)
    if data_key_encrypted is None:
        return False
    logging.info('Created new AWS KMS data key')

    f = Fernet(data_key_plaintext)
    file_contents_encrypted = f.encrypt(file_contents)

    try:
        with open(filename[:-len(filename.split('.')[-1])] + 'encrypted.'+filename.split('.')[-1], 'wb') as file_encrypted:
            file_encrypted.write(len(data_key_encrypted).to_bytes(NUM_BYTES_FOR_LEN,
                                                                  byteorder='big'))
            file_encrypted.write(data_key_encrypted)
            file_encrypted.write(file_contents_encrypted)
    except IOError as e:
        logging.error(e)
        return False

    return True



def decrypt_file(filename):
    """Decrypt a file encrypted by encrypt_file()
    The encrypted file is read from <filename>.encrypted
    The decrypted file is written to <filename>.decrypted
    :param filename: File to decrypt
    :return: True if file was decrypted. Otherwise, False.
    """

    try:
        with open(filename[:-len(filename.split('.')[-1])] + 'encrypted.'+filename.split('.')[-1], 'rb') as file:
            file_contents = file.read()
    except IOError as e:
        logging.error(e)
        return False

    data_key_encrypted_len = int.from_bytes(file_contents[:NUM_BYTES_FOR_LEN],
                                            byteorder='big') \
                             + NUM_BYTES_FOR_LEN
    data_key_encrypted = file_contents[NUM_BYTES_FOR_LEN:data_key_encrypted_len]

    data_key_plaintext = decrypt_data_key(data_key_encrypted)
    if data_key_plaintext is None:
        logging.error("Cannot decrypt the data key")
        return False
    f = Fernet(data_key_plaintext)
    file_contents_decrypted = f.decrypt(file_contents[data_key_encrypted_len:])

    try:
        with open(filename[:-len(filename.split('.')[-1])] + 'decrypted.'+filename.split('.')[-1] , 'wb') as file_decrypted:
            file_decrypted.write(file_contents_decrypted)
    except IOError as e:
        logging.error(e)
        return False

    return True



def encrypt_file_desc(file_to_encrypt, cmk_description):
    """Exercise AWS KMS operations retrieve_cmk(), create_cmk(),
    create_data_key(), and decrypt_data_key().
    Also, use the various KMS keys to encrypt and decrypt a file.
    """
    
    cmk_description = 'Key for file encryption'

    cmk_id, cmk_arn = retrieve_cmk(cmk_description)
    if cmk_id is None:
        cmk_id, cmk_arn = create_cmk(cmk_description)
        if cmk_id is None:
            exit(1)
        
    if file_to_encrypt:
        if encrypt_file(file_to_encrypt, cmk_arn):
            print('encrypt done')
