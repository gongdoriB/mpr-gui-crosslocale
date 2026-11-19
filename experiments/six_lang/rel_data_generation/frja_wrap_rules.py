FRJA_WRAP = {
    'fr': (r'^(?:(?:Bouton|bouton|Option|option|Icône|icône|Section|section|Étiquette|étiquette|Onglet|onglet|Lien|lien|Champ|champ|Case à cocher|Texte|texte|Menu déroulant|Interrupteur|Titre de la section)(?:\s+(?:de la |du |des |de l\'|de |d\'))?\s*)',
           r'\s+(?:bouton|onglet|texte|icône|section|lien|étiquette)$'),
    'ja': (r'^', r'(?:のテキスト|カウントダウンテキスト|セクションタイトル|トグルスイッチ|チェックボックス|ドロップダウン|ボタン|アイコン|タブ|リンク|フィールド|オプション|テキスト|ラベル|セクション|進行バー|チャレンジ)$'),
}
