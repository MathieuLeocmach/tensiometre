BOOL err; 
  DWORD ret_bytes; 
  CEdit*  pE; 
  char Pos[6]; 
  CString s; 


//    unsigned char txbuf[3], rxbuf[3]; 
char txbuf[13], rxbuf[89]; 

      int i; 
  BYTE  nikon_para; 

m_bCycled=true; 

  if (m_bCycled==false) 
  { 

      txbuf[0] = 'I'; 
      txbuf[1]=1; 

//////        txbuf[6]=0x7D; 
      Write(txbuf, 2, &ret_bytes); 
      Read(rxbuf, 2, &ret_bytes); 

      Sleep(100); 

      txbuf[0] = 'I'; 
      txbuf[1]=2; 

//////        txbuf[6]=0x7D; 
      Write(txbuf, 2, &ret_bytes); 
      Read(rxbuf, 2, &ret_bytes); 

      Sleep(100); 


      txbuf[0] = 'I'; 
      txbuf[1]=1; 

//////        txbuf[6]=0x7D; 
      Write(txbuf, 2, &ret_bytes); 
      Read(rxbuf, 2, &ret_bytes); 

      Sleep(100); 

      m_bCycled=true; 
  } 

  if (m_bCycled) 
  { 


for (i=0;i<13;i++) 
      txbuf[i]=0; 

      Sleep(30); 
      txbuf[0] = 'F';             //total no.TEST S 
      Write(txbuf, 1, &ret_bytes); 

      Sleep(30); 
      txbuf[0] = 'S';             //total no.TEST S 
      Write(txbuf, 1, &ret_bytes); 
      Sleep(30); 

      txbuf[0]=nikon_para=0x0F; 
//        Write((void *)nikon_para, 1, &ret_bytes); 
      Write(txbuf, 1, &ret_bytes); 
      Sleep(30); 


  pE=(CEdit*)GetDlgItem(IDC_EDIT_NEWX); 
  pE->GetLine(0,Pos); 
  m_nNewX=atoi(Pos); 
  x=m_nNewX*16;
  
  pE=(CEdit*)GetDlgItem(IDC_EDIT_NEWY); 
  pE->GetLine(0,Pos); 
  m_nNewY=atoi(Pos); 
  y=m_nNewY*16; 
  
  pE=(CEdit*)GetDlgItem(IDC_EDIT_NEWZ); 
  pE->GetLine(0,Pos); 
  m_nNewZ=atoi(Pos); 
  z=m_nNewZ*16; 

  

//    if ((nikon_para & 0x10) != 0) 
//    {         for (i=0;i<13;i++) 
      txbuf[i]=0; 
  memcpy(txbuf+0,(char*)&x,4); 
  memcpy(txbuf+4,(char*)&y,4); 
  memcpy(txbuf+8,(char*)&z,4); 


  ResetDevice(); 
      Purge(FT_PURGE_RX || FT_PURGE_TX); 
      ResetDevice(); 
      SetTimeouts(3000, 3000);//extend timeout while board finishes reset 
      Sleep(20); 
      Write(txbuf, 12, &ret_bytes); 
//    } 
  } 

      Read(rxbuf, 1, &ret_bytes); 
      Sleep(2); 
