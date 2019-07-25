from __future__ import print_function
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from httplib2 import Http
from oauth2client import file, client, tools
from apiclient.http import MediaFileUpload,MediaIoBaseDownload
import io

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive']

class GoogleDrive():
    def __init__(self):
        self.service = None

    def connect(self):
        """Shows basic usage of the Drive v3 API.
        Prints the names and ids of the first 10 files the user has access to.
        """
        creds = None
        # The file token.pickle stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)

        self.service = build('drive', 'v3', credentials=creds)

    def createFolder(self, folderName, parentID=None):
        body = {
            'name': folderName,
            'mimeType': "application/vnd.google-apps.folder"
        }
        if parentID:
            body['parents'] = [parentID]
        root_folder = self.service.files().create(body = body).execute()
        ID = root_folder.get('id')
        return ID

    def uploadFile(self, filePath, fileName, parentID=None):
        file_metadata = {
            'name': fileName,
            'mimeType': '*/*'
        }
        if parentID:
            file_metadata['parents'] = [parentID]
        media = MediaFileUpload(filePath,
                                mimetype='*/*',
                                resumable=True)
        file = self.service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        # print ('File ID: ' + file.get('id'))
        return file.get('id')

    def listFiles(self, numFile=10):
        # Call the Drive v3 API
        results = self.service.files().list(
            pageSize=numFile, fields="nextPageToken, files(id, name)").execute()
        items = results.get('files', [])

        # if not items:
        #     print('No files found.')
        # else:
        #     print('Files:')
        #     for item in items:
        #         print(u'{0} ({1})'.format(item['name'], item['id']))
        return items
    
    def searchFolder(self, folderName, numFile=1):
        # Call the Drive v3 API
        results = self.service.files().list(
            q="mimeType='application/vnd.google-apps.folder' and name='{}'".format(folderName),
            pageSize=numFile, 
            fields="nextPageToken, files(id, name)").execute()
        items = results.get('files', [])

        if not items:
            # print('No files found.')
            return None
        else:
            return items[0]['id']
        #     print('Files:')
        #     for item in items:
        #         print(u'{0} ({1})'.format(item['name'], item['id']))
        # return items
    