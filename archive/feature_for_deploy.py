# -*- coding: utf-8 -*-

from pydantic import BaseModel
import json

class DATA(BaseModel):
    trackID: int
    title: str
    tags: str
    loudness: float
    tempo:  float
    time_signature: int
    key: int
    mode: int
    duration: float
    vect_1: float
    vect_2: float
    vect_3 :float
    vect_4 :float
    vect_5 :float
    vect_6 :float
    vect_7 :float
    vect_8 :float
    vect_9 :float
    vect_10:float
    vect_11:float
    vect_12:float
    vect_13:float
    vect_14:float
    vect_15:float
    vect_16:float
    vect_17:float
    vect_18:float
    vect_19:float
    vect_20:float
    vect_21:float
    vect_22:float
    vect_23:float
    vect_24:float
    vect_25:float
    vect_26:float
    vect_27:float
    vect_28:float
    vect_29:float
    vect_30:float
    vect_31:float
    vect_32:float
    vect_33:float
    vect_34:float
    vect_35:float
    vect_36:float
    vect_37:float
    vect_38:float
    vect_39:float
    vect_40:float
    vect_41:float
    vect_42:float
    vect_43:float
    vect_44:float
    vect_45:float
    vect_46:float
    vect_47:float
    vect_48:float
    vect_49:float
    vect_50:float
    vect_51:float
    vect_52:float
    vect_53:float
    vect_54:float
    vect_55:float
    vect_56:float
    vect_57:float
    vect_58:float
    vect_59:float
    vect_60:float
    vect_61:float
    vect_62:float
    vect_63:float
    vect_64:float
    vect_65:float
    vect_66:float
    vect_67:float
    vect_68:float
    vect_69:float
    vect_70:float
    vect_71:float
    vect_72:float
    vect_73:float
    vect_74:float
    vect_75:float
    vect_76:float
    vect_77:float
    vect_78:float
    vect_79:float
    vect_80:float
    vect_81:float
    vect_82:float
    vect_83:float
    vect_84:float
    vect_85:float
    vect_86:float
    vect_87:float
    vect_88:float
    vect_89:float
    vect_90:float
    vect_91:float
    vect_92:float
    vect_93:float
    vect_94:float
    vect_95:float
    vect_96:float
    vect_97:float
    vect_98:float
    vect_99:float
    vect_100:float
    vect_101:float
    vect_102:float
    vect_103:float
    vect_104:float
    vect_105:float
    vect_106:float
    vect_107:float
    vect_108:float
    vect_109:float
    vect_110:float
    vect_111:float
    vect_112:float
    vect_113:float
    vect_114:float
    vect_115:float
    vect_116:float
    vect_117:float
    vect_118:float
    vect_119:float
    vect_120:float
    vect_121:float
    vect_122:float
    vect_123:float
    vect_124:float
    vect_125:float
    vect_126:float
    vect_127:float
    vect_128:float
    vect_129:float
    vect_130:float
    vect_131:float
    vect_132:float
    vect_133:float
    vect_134:float
    vect_135:float
    vect_136:float
    vect_137:float
    vect_138:float
    vect_139:float
    vect_140:float
    vect_141:float
    vect_142:float
    vect_143:float
    vect_144:float
    vect_145:float
    vect_146:float
    vect_147:float
    vect_148:float

#
#
# data={}
#     trackID = data['trackID']
#     title=data['title']
#     tags =data['tags']
#     loudness =data['loudness']
#     tempo =data['tempo ']
#     time_signature  =data['time_signature']
#     key = data['key']
#     mode  =data['mode']
#     duration=data['duration']
#     vect_1 =data['vect_1']
#     vect_2 =data['vect_2']
#     vect_3 =data['vect_3']
#     vect_4 =data['vect_4']
#     vect_5 =data['vect_5']
#     vect_6 =data['vect_6']
#     vect_7 =data['vect_7']
#     vect_8 =data['vect_8']
#     vect_9 =data['vect_9']
#     vect_10=data['vect_10']
#     vect_11=data['vect_11']
#     vect_12=data['vect_12']
#     vect_13=data['vect_13']
#     vect_14=data['vect_14']
#     vect_15=data['vect_15']
#     vect_16=data['vect_16']
#     vect_17=data['vect_17']
#     vect_18=data['vect_18']
#     vect_19=data['vect_19']
#     vect_20=data['vect_20']
#     vect_21=data['vect_21']
#     vect_22=data['vect_22']
#     vect_23=data['vect_23']
#     vect_24=data['vect_24']
#     vect_25=data['vect_25']
#     vect_26=data['vect_26']
#     vect_27=data['vect_27']
#     vect_28=data['vect_28']
#     vect_29=data['vect_29']
#     vect_30=data['vect_30']
#     vect_31=data['vect_31']
#     vect_32=data['vect_32']
#     vect_33=data['vect_33']
#     vect_34=data['vect_34']
#     vect_35=data['vect_35']
#     vect_36=data['vect_36']
#     vect_37=data['vect_37']
#     vect_38=data['vect_38']
#     vect_39=data['vect_39']
#     vect_40=data['vect_40']
#     vect_41=data['vect_41']
#     vect_42=data['vect_42']
#     vect_43=data['vect_43']
#     vect_44=data['vect_44']
#     vect_45=data['vect_45']
#     vect_46=data['vect_46']
#     vect_47=data['vect_47']
#     vect_48=data['vect_48']
#     vect_49=data['vect_49']
#     vect_50=data['vect_50']
#     vect_51=data['vect_51']
#     vect_52=data['vect_52']
#     vect_53=data['vect_53']
#     vect_54=data['vect_54']
#     vect_55=data['vect_55']
#     vect_56=data['vect_56']
#     vect_57=data['vect_57']
#     vect_58=data['vect_58']
#     vect_59=data['vect_59']
#     vect_60=data['vect_60']
#     vect_61=data['vect_61']
#     vect_62=data['vect_62']
#     vect_63=data['vect_63']
#     vect_64=data['vect_64']
#     vect_65=data['vect_65']
#     vect_66=data['vect_66']
#     vect_67=data['vect_67']
#     vect_68=data['vect_68']
#     vect_69=data['vect_69']
#     vect_70=data['vect_70']
#     vect_71=data['vect_71']
#     vect_72=data['vect_72']
#     vect_73=data['vect_73']
#     vect_74=data['vect_74']
#     vect_75=data['vect_75']
#     vect_76=data['vect_76']
#     vect_77=data['vect_77']
#     vect_78=data['vect_78']
#     vect_79=data['vect_79']
#     vect_80=data['vect_80']
#     vect_81=data['vect_81']
#     vect_82=data['vect_82']
#     vect_83=data['vect_83']
#     vect_84=data['vect_84']
#     vect_85=data['vect_85']
#     vect_86=data['vect_86']
#     vect_87=data['vect_87']
#     vect_88=data['vect_88']
#     vect_89=data['vect_89']
#     vect_90=data['vect_90']
#     vect_91=data['vect_91']
#     vect_92=data['vect_92']
#     vect_93=data['vect_93']
#     vect_94=data['vect_94']
#     vect_95=data['vect_95']
#     vect_96=data['vect_96']
#     vect_97=data['vect_97']
#     vect_98=data['vect_98']
#     vect_99=data['vect_99']
#     vect_100=data['vect_100']
#     vect_101=data['vect_101']
#     vect_102=data['vect_102']
#     vect_103=data['vect_103']
#     vect_104=data['vect_104']
#     vect_105=data['vect_105']
#     vect_106=data['vect_106']
#     vect_107=data['vect_107']
#     vect_108=data['vect_108']
#     vect_109=data['vect_109']
#     vect_110=data['vect_110']
#     vect_111=data['vect_111']
#     vect_112=data['vect_112']
#     vect_113=data['vect_113']
#     vect_114=data['vect_114']
#     vect_115=data['vect_115']
#     vect_116=data['vect_116']
#     vect_117=data['vect_117']
#     vect_118=data['vect_118']
#     vect_119=data['vect_119']
#     vect_120=data['vect_120']
#     vect_121=data['vect_121']
#     vect_122=data['vect_122']
#     vect_123=data['vect_123']
#     vect_124=data['vect_124']
#     vect_125=data['vect_125']
#     vect_126=data['vect_126']
#     vect_127=data['vect_127']
#     vect_128=data['vect_128']
#     vect_129=data['vect_129']
#     vect_130=data['vect_130']
#     vect_131=data['vect_131']
#     vect_132=data['vect_132']
#     vect_133=data['vect_133']
#     vect_134=data['vect_134']
#     vect_135=data['vect_135']
#     vect_136=data['vect_136']
#     vect_137=data['vect_137']
#     vect_138=data['vect_138']
#     vect_139=data['vect_139']
#     vect_140=data['vect_140']
#     vect_141=data['vect_141']
#     vect_142=data['vect_142']
#     vect_143=data['vect_143']
#     vect_144=data['vect_144']
#     vect_145=data['vect_145']
#     vect_146=data['vect_146']
#     vect_147=data['vect_147']
#           vect_148=data['vect_148']


